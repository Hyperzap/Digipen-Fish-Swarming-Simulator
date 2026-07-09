using Unity.Collections;
using Unity.Mathematics;
using FishSwarm.Behavior;
using Random = Unity.Mathematics.Random;

namespace FishSwarm.Jobs
{
    /// <summary>
    /// Burst-compatible per-fish simulation, written with <c>Unity.Mathematics</c> and only
    /// blittable value/native types. Every method here runs Burst-compiled + parallel when called
    /// from a scheduled <c>[BurstCompile]</c> job (Burst ON), and as plain single-threaded IL when
    /// called from a main-thread loop (Burst OFF) — one implementation, honest comparison.
    ///
    /// Neighbour data lives in a <see cref="NativeParallelMultiHashMap{TKey,TValue}"/> keyed by a
    /// collision-free cell index (grid is bounded to the ocean volume, cell size = perception
    /// radius, so the 3×3×3 neighbourhood covers the radius).
    /// </summary>
    public static class FishSteering
    {
        // ---- Grid cell helpers -------------------------------------------------------------

        public static int3 CellCoords(float3 pos, in SteeringParams p)
        {
            int3 c = (int3)math.floor((pos - p.gridOrigin) / p.cellSize);
            return math.clamp(c, int3.zero, p.gridDim - 1);
        }

        public static int CellKey(int3 c, in SteeringParams p)
        {
            return c.x + p.gridDim.x * (c.y + p.gridDim.y * c.z);
        }

        // ---- Neighbour accumulation --------------------------------------------------------

        struct Accum
        {
            public float3 sep;
            public float3 aliSum;
            public float3 cohSum;
            public int flock;
            public float nearestSqr;
        }

        static bool AddNeighbor(ref Accum a, float3 myPos, int j,
            in NativeArray<float3> pos, in NativeArray<float3> vel,
            float perSqr, float sepSqr, int maxN)
        {
            float3 off = myPos - pos[j];
            float d2 = math.lengthsq(off);
            if (d2 < perSqr && d2 > 1e-6f)
            {
                if (d2 < a.nearestSqr) a.nearestSqr = d2;
                if (d2 < sepSqr) a.sep += off / d2;
                a.aliSum += vel[j];
                a.cohSum += pos[j];
                a.flock++;
                if (a.flock >= maxN) return true; // cap reached
            }
            return false;
        }

        static bool ScanCell(ref Accum a, int key, float3 myPos, int self,
            NativeParallelMultiHashMap<int, int> grid,
            in NativeArray<float3> pos, in NativeArray<float3> vel,
            float perSqr, float sepSqr, int maxN)
        {
            if (grid.TryGetFirstValue(key, out int j, out var it))
            {
                do
                {
                    if (j != self && AddNeighbor(ref a, myPos, j, pos, vel, perSqr, sepSqr, maxN))
                        return true;
                }
                while (grid.TryGetNextValue(out j, ref it));
            }
            return false;
        }

        // ---- Steering forces ---------------------------------------------------------------

        static float3 ClampMag(float3 v, float max)
        {
            float m2 = math.lengthsq(v);
            if (m2 > max * max) return v * (max / math.sqrt(m2));
            return v;
        }

        // Reynolds steer: full-speed desired toward dir, minus current velocity, clamped.
        static float3 Steer(float3 dir, float3 vel, in SteeringParams p)
        {
            if (math.lengthsq(dir) < 1e-8f) return float3.zero;
            float3 desired = math.normalize(dir) * p.maxSpeed;
            return ClampMag(desired - vel, p.maxSteerForce);
        }

        static float3 Boundary(float3 pos, float3 vel, in SteeringParams p)
        {
            float3 mn = p.boundsCenter - p.boundsSize * 0.5f;
            float3 mx = p.boundsCenter + p.boundsSize * 0.5f;
            float m = p.boundaryMargin;

            float3 inward = float3.zero;
            if (pos.x < mn.x + m) inward.x += 1f; else if (pos.x > mx.x - m) inward.x -= 1f;
            if (pos.y < mn.y + m) inward.y += 1f; else if (pos.y > mx.y - m) inward.y -= 1f;
            if (pos.z < mn.z + m) inward.z += 1f; else if (pos.z > mx.z - m) inward.z -= 1f;

            if (math.lengthsq(inward) < 1e-8f) return float3.zero;
            return Steer(inward, vel, p) * p.boundaryWeight;
        }

        static float3 Threat(float3 pos, float3 vel,
            in NativeArray<float3> predators, int predatorCount, in SteeringParams p, out byte inRange)
        {
            inRange = 0;
            float3 flee = float3.zero;
            float r = p.predatorFleeRadius;
            float r2 = r * r;
            for (int k = 0; k < predatorCount; k++)
            {
                float3 off = pos - predators[k];
                float d2 = math.lengthsq(off);
                if (d2 < r2 && d2 > 1e-6f)
                {
                    inRange = 1;
                    float d = math.sqrt(d2);
                    flee += (off / d) * (1f - d / r); // away, ramps up as predator nears
                }
            }
            if (math.lengthsq(flee) < 1e-8f) return float3.zero;
            return Steer(flee, vel, p) * p.threatWeight;
        }

        static int NearestNode(float3 pos, in NativeArray<float3> nodes, int nodeCount, in SteeringParams p)
        {
            float best = p.foragingNodeRadius * p.foragingNodeRadius;
            int nearest = -1;
            for (int k = 0; k < nodeCount; k++)
            {
                float d2 = math.lengthsq(nodes[k] - pos);
                if (d2 < best) { best = d2; nearest = k; }
            }
            return nearest;
        }

        // ---- Per-fish steering step --------------------------------------------------------

        public static void Step(int i,
            in NativeArray<float3> positions, in NativeArray<float3> velocities,
            in NativeArray<FishState> states,
            in NativeArray<float3> predators, int predatorCount,
            in NativeArray<float3> nodes, int nodeCount,
            NativeParallelMultiHashMap<int, int> grid, bool useGrid,
            in SteeringParams p,
            NativeArray<float3> accelerations,
            NativeArray<float> nearestNeighborSqr,
            NativeArray<byte> predatorInRange)
        {
            float3 myPos = positions[i];
            float3 myVel = velocities[i];
            int state = (int)states[i];

            float perSqr = p.perceptionRadius * p.perceptionRadius;
            float sepSqr = p.separationRadius * p.separationRadius;
            int maxN = p.maxNeighbors;

            Accum a = new Accum { nearestSqr = float.MaxValue };

            if (useGrid)
            {
                // Own cell first (closest, symmetric), then the 26 neighbours; early-out at the cap.
                int3 cc = CellCoords(myPos, p);
                bool done = ScanCell(ref a, CellKey(cc, p), myPos, i, grid, positions, velocities, perSqr, sepSqr, maxN);
                for (int dz = -1; dz <= 1 && !done; dz++)
                for (int dy = -1; dy <= 1 && !done; dy++)
                for (int dx = -1; dx <= 1 && !done; dx++)
                {
                    if ((dx | dy | dz) == 0) continue; // centre already scanned
                    int3 nc = cc + new int3(dx, dy, dz);
                    if (math.any(nc < 0) || math.any(nc >= p.gridDim)) continue;
                    done = ScanCell(ref a, CellKey(nc, p), myPos, i, grid, positions, velocities, perSqr, sepSqr, maxN);
                }
            }
            else
            {
                int count = positions.Length;
                for (int j = 0; j < count; j++)
                {
                    if (j == i) continue;
                    if (AddNeighbor(ref a, myPos, j, positions, velocities, perSqr, sepSqr, maxN))
                        break;
                }
            }

            float3 sac = p.SacFor(state);
            float3 force = float3.zero;

            if (a.flock > 0)
            {
                float3 avgVel = a.aliSum / a.flock;
                force += Steer(avgVel, myVel, p) * (p.alignmentWeight * sac.y);

                // Anisotropic cohesion: scale the force per axis so the school is ellipsoidal.
                float3 toCentre = (a.cohSum / a.flock) - myPos;
                float3 coh = Steer(toCentre, myVel, p) * p.cohesionAxisWeights;
                force += coh * (p.cohesionWeight * sac.z);
            }

            if (math.lengthsq(a.sep) > 1e-8f)
                force += Steer(a.sep, myVel, p) * (p.separationWeight * sac.x);

            force += Boundary(myPos, myVel, p);
            force += Threat(myPos, myVel, predators, predatorCount, p, out byte inRange);

            // Foraging pull toward the nearest node while in the Foraging state.
            if (state == (int)FishState.Foraging)
            {
                int node = NearestNode(myPos, nodes, nodeCount, p);
                if (node >= 0)
                    force += Steer(nodes[node] - myPos, myVel, p) * p.foragingWeight;
            }

            accelerations[i] = force;
            nearestNeighborSqr[i] = a.nearestSqr;
            predatorInRange[i] = inRange;
        }

        // ---- Integration -------------------------------------------------------------------

        public static void Integrate(int i,
            NativeArray<float3> positions, NativeArray<float3> velocities,
            in NativeArray<float3> accelerations, NativeArray<float4x4> matrices,
            float dt, in SteeringParams p, float3 scale)
        {
            float3 v = velocities[i] + accelerations[i] * dt;
            float speed = math.length(v);
            if (speed < 1e-5f)
            {
                float3 prev = velocities[i];
                v = (math.lengthsq(prev) > 1e-10f ? math.normalize(prev) : new float3(0f, 0f, 1f)) * p.minSpeed;
            }
            else
            {
                v = v / speed * math.clamp(speed, p.minSpeed, p.maxSpeed);
            }

            float3 pos = positions[i] + v * dt;
            velocities[i] = v;
            positions[i] = pos;

            quaternion rot = quaternion.LookRotationSafe(v, math.up());
            matrices[i] = float4x4.TRS(pos, rot, scale);
        }

        // ---- Behaviour tick (staggered) ----------------------------------------------------

        public static void TickBehavior(int i,
            in NativeArray<float3> positions, in NativeArray<float> nearestNeighborSqr,
            in NativeArray<byte> predatorInRange, in NativeArray<float3> nodes, int nodeCount,
            NativeArray<FishState> states, NativeArray<float> stateTimer,
            in SteeringParams p, float tickDt, uint frameSeed)
        {
            bool nearNode = NearestNode(positions[i], nodes, nodeCount, p) >= 0;
            float nnDist = math.sqrt(nearestNeighborSqr[i]);

            var rng = new Random((uint)(i * 747796405) ^ frameSeed | 1u);
            float timer = stateTimer[i];

            states[i] = FishBehaviorController.Tick(
                states[i],
                predatorInRange[i] != 0,
                nnDist,
                nearNode,
                rng.NextFloat(),
                p.regroupNeighborDistance,
                p.foragingChancePerTick,
                p.foragingDuration,
                tickDt,
                ref timer);

            stateTimer[i] = timer;
        }

        // ---- Colour bucketing (counting sort by state) -------------------------------------

        /// <summary>
        /// Reorder <paramref name="matrices"/> into <paramref name="sorted"/> grouped by state, and
        /// write each state's start offset + count so rendering can draw one colour per contiguous
        /// run. Two O(n) passes, single-threaded.
        /// </summary>
        public static void BucketByState(int count,
            in NativeArray<FishState> states, in NativeArray<float4x4> matrices,
            NativeArray<float4x4> sorted, NativeArray<int> stateStart, NativeArray<int> stateCount)
        {
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (int i = 0; i < count; i++)
            {
                switch ((int)states[i])
                {
                    case 1: c1++; break;
                    case 2: c2++; break;
                    case 3: c3++; break;
                    default: c0++; break;
                }
            }

            int s0 = 0, s1 = s0 + c0, s2 = s1 + c1, s3 = s2 + c2;
            stateStart[0] = s0; stateStart[1] = s1; stateStart[2] = s2; stateStart[3] = s3;
            stateCount[0] = c0; stateCount[1] = c1; stateCount[2] = c2; stateCount[3] = c3;

            int w0 = s0, w1 = s1, w2 = s2, w3 = s3;
            for (int i = 0; i < count; i++)
            {
                switch ((int)states[i])
                {
                    case 1: sorted[w1++] = matrices[i]; break;
                    case 2: sorted[w2++] = matrices[i]; break;
                    case 3: sorted[w3++] = matrices[i]; break;
                    default: sorted[w0++] = matrices[i]; break;
                }
            }
        }
    }
}
