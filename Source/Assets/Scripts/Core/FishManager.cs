using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using FishSwarm.Flocking;
using FishSwarm.Spatial;
using FishSwarm.Behavior;
using FishSwarm.Predators;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace FishSwarm.Core
{
    /// <summary>
    /// Owns the whole school as struct-of-arrays data (positions/velocities/state) and steps every
    /// fish in one central loop. This data-oriented layout — rather than one MonoBehaviour per
    /// fish — is what ports cleanly to Burst <c>IJobParallelFor</c> + <c>NativeArray</c> in the
    /// scale-up pass: only the loop body moves into a job.
    ///
    /// Neighbour lookups go through a pluggable <see cref="INeighborProvider"/> so the same live
    /// population can flip between the naive O(n²) scan and the spatial grid at runtime, with the
    /// lookup cost isolated by a <see cref="Stopwatch"/> for the benchmark.
    ///
    /// Behaviour: a threat force pushes fish away from predators every frame (the split emerges
    /// from near-vs-far fish reacting differently), while a low-frequency, staggered state machine
    /// (Schooling/Fleeing/Regrouping/Foraging) modulates the cohesion weights and colours the fish.
    ///
    /// Fish are drawn with GPU instancing, bucketed by state so each state gets its own colour in a
    /// few instanced draw calls — no per-fish GameObject or Transform.
    /// </summary>
    public class FishManager : MonoBehaviour
    {
        const int StateCount = 4;

        [Header("References (auto-resolved if left empty)")]
        [Tooltip("Shared steering parameters. A runtime default is used if none is assigned.")]
        public FishConfig config;
        [Tooltip("Ocean volume the school lives in. Falls back to any OceanBounds in the scene.")]
        public OceanBounds bounds;

        [Header("Population")]
        [Min(0)] public int spawnCount = 300;
        [Tooltip("Deterministic spawn layout when set; use 0 for a fresh random layout each run.")]
        public int randomSeed = 12345;
        [Tooltip("Fish spawn within this radius of the volume centre so they start as one " +
                 "coherent school. Use <= 0 to scatter them across the whole volume.")]
        public float spawnRadius = 15f;

        [Header("Fish appearance")]
        [Tooltip("Fish material asset (URP/Lit with GPU Instancing enabled). Assigning a real asset " +
                 "is what keeps the instancing shader variant from being stripped out of builds — " +
                 "without it, RenderMeshInstanced silently draws nothing in a standalone player.")]
        public Material fishMaterial;
        [Tooltip("Fallback shader if no material asset is assigned (editor convenience only).")]
        public Shader fishShader;
        public Vector3 fishScale = new Vector3(0.25f, 0.25f, 0.7f);
        [Tooltip("Colour per state: 0 Schooling, 1 Fleeing, 2 Regrouping, 3 Foraging.")]
        public Color[] stateColors =
        {
            new Color(0.80f, 0.85f, 0.55f), // Schooling
            new Color(0.90f, 0.20f, 0.15f), // Fleeing
            new Color(0.20f, 0.60f, 0.95f), // Regrouping
            new Color(0.30f, 0.85f, 0.35f), // Foraging
        };

        [Header("Neighbour lookup")]
        [Tooltip("Start with the spatial grid (fast). Toggle to naive at runtime to see the cost.")]
        public bool useGridAtStart = true;
        [Tooltip("Key that flips between naive O(n²) and the spatial grid on the live population.")]
        public Key toggleModeKey = Key.Tab;
        [Tooltip("Minimal on-screen stats. A stopgap until the full benchmark overlay is built.")]
        public bool showStats = true;

        // Struct-of-arrays simulation state.
        Vector3[] _positions;
        Vector3[] _velocities;
        Vector3[] _accelerations;
        FishState[] _states;
        float[] _stateTimer;         // time spent in the current state (for foraging duration)
        float[] _nearestNeighborSqr; // per-frame, for the Regrouping -> Schooling test
        bool[] _predatorInRange;     // per-frame, for the -> Fleeing trigger
        int _count;

        // Reused per-fish neighbour scratch buffer (no per-frame allocation).
        readonly List<int> _neighbors = new List<int>(256);

        // Predator / foraging-node positions snapshotted each frame.
        Vector3[] _predatorPositions = new Vector3[8];
        int _predatorCount;
        Vector3[] _nodePositions = new Vector3[8];
        int _nodeCount;

        // Pluggable neighbour lookup + isolated timing.
        NaiveNeighborProvider _naive;
        UniformGridNeighborProvider _grid;
        INeighborProvider _active;
        readonly Stopwatch _queryTimer = new Stopwatch();
        readonly Stopwatch _simTimer = new Stopwatch();
        double _lastQueryMs;
        double _lastSimMs;

        // Per-state instanced rendering.
        Mesh _fishMesh;
        Material[] _stateMaterials;
        RenderParams[] _stateRenderParams;
        Matrix4x4[][] _stateMatrices; // [state][instance]
        int[] _stateCounts;

        public int Count => _count;
        public string ActiveModeName => _active != null ? _active.Name : "-";
        public double LastNeighborQueryMs => _lastQueryMs;
        public double LastSimMs => _lastSimMs;

        /// <summary>Flip between the naive and grid providers on the live population.</summary>
        public void ToggleNeighborMode()
        {
            _active = ReferenceEquals(_active, _grid) ? (INeighborProvider)_naive : _grid;
        }

        void Start()
        {
            if (config == null)
                config = FishConfig.CreateDefault();
            if (bounds == null)
                bounds = FindFirstObjectByType<OceanBounds>();
            if (bounds == null)
            {
                Debug.LogError("[FishManager] No OceanBounds found in the scene; cannot spawn.");
                enabled = false;
                return;
            }

            _naive = new NaiveNeighborProvider();
            _grid = new UniformGridNeighborProvider();
            _active = useGridAtStart ? (INeighborProvider)_grid : _naive;

            BuildRenderResources();
            Spawn(spawnCount);
        }

        void BuildRenderResources()
        {
            // Borrow a built-in cube mesh, then discard the temporary primitive (and its collider).
            var temp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _fishMesh = temp.GetComponent<MeshFilter>().sharedMesh;
            Destroy(temp);

            // Prefer a real material asset: it's what keeps the GPU-instancing shader variant in the
            // build. We clone one tinted material per state so states are colour-coded on screen.
            Material baseMat = fishMaterial;
            bool tempBase = false;
            if (baseMat == null)
            {
                Shader shader = fishShader != null ? fishShader : Shader.Find("Universal Render Pipeline/Lit");
                if (shader == null) shader = Shader.Find("Standard");
                if (shader == null)
                    Debug.LogError("[FishManager] No fish material/shader found; fish will be invisible.");
                baseMat = new Material(shader);
                tempBase = true;
            }

            Bounds b = bounds.Bounds;
            var worldBounds = new Bounds(b.center, b.size + Vector3.one * 10f);

            _stateMaterials = new Material[StateCount];
            _stateRenderParams = new RenderParams[StateCount];
            for (int s = 0; s < StateCount; s++)
            {
                var m = new Material(baseMat) { color = ColorForState(s) };
                m.enableInstancing = true;
                _stateMaterials[s] = m;
                _stateRenderParams[s] = new RenderParams(m)
                {
                    worldBounds = worldBounds,
                    shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
                    receiveShadows = false,
                };
            }

            if (tempBase) Destroy(baseMat);
        }

        Color ColorForState(int s)
        {
            if (stateColors != null && s < stateColors.Length) return stateColors[s];
            return Color.white;
        }

        void Spawn(int n)
        {
            _count = Mathf.Max(0, n);
            _positions = new Vector3[_count];
            _velocities = new Vector3[_count];
            _accelerations = new Vector3[_count];
            _states = new FishState[_count];
            _stateTimer = new float[_count];
            _nearestNeighborSqr = new float[_count];
            _predatorInRange = new bool[_count];

            _stateMatrices = new Matrix4x4[StateCount][];
            _stateCounts = new int[StateCount];
            for (int s = 0; s < StateCount; s++)
                _stateMatrices[s] = new Matrix4x4[_count];

            Random.State prev = Random.state;
            if (randomSeed != 0) Random.InitState(randomSeed);

            float startSpeed = 0.5f * (config.minSpeed + config.maxSpeed);
            Vector3 centre = bounds.Bounds.center;
            for (int i = 0; i < _count; i++)
            {
                _positions[i] = spawnRadius > 0f
                    ? bounds.Clamp(centre + Random.insideUnitSphere * spawnRadius)
                    : bounds.RandomPointInside();
                _velocities[i] = Random.onUnitSphere * startSpeed;
                _states[i] = FishState.Schooling;
            }

            Random.state = prev;
        }

        void Update()
        {
            if (_count == 0) return;

            HandleInput();
            SnapshotThreatsAndNodes();

            float dt = Time.deltaTime;
            Bounds volume = bounds.Bounds;

            // Time the whole CPU simulation so the overlay can separate it from GPU/present time.
            _simTimer.Restart();

            _queryTimer.Reset();
            _queryTimer.Start();
            _active.Build(_positions, _count, config.perceptionRadius, config.maxNeighbors);
            _queryTimer.Stop();

            // Pass 1 — accumulate accelerations from a snapshot of current state. Only the neighbour
            // query is inside the query stopwatch; steering maths stays outside it.
            for (int i = 0; i < _count; i++)
            {
                _queryTimer.Start();
                _active.Query(i, _neighbors);
                _queryTimer.Stop();

                Vector3 pos = _positions[i];
                Vector3 vel = _velocities[i];

                Vector3 accel = BoidSteering.Compute(i, _positions, _velocities, _neighbors, config, config.SacFor(_states[i]));
                accel += BoundarySteering.Compute(pos, vel, volume, config);
                accel += ThreatSteering.Compute(pos, vel, _predatorPositions, _predatorCount, config, out bool inRange);

                _predatorInRange[i] = inRange;
                _nearestNeighborSqr[i] = NearestNeighborSqr(i);

                if (_states[i] == FishState.Foraging)
                    accel += ForagingPull(pos, vel);

                _accelerations[i] = accel;
            }
            _lastQueryMs = _queryTimer.Elapsed.TotalMilliseconds;

            // Low-frequency, staggered state machine.
            TickBehaviors(dt);

            // Pass 2 — integrate, clamp speed, and bucket each fish's matrix by state for rendering.
            for (int s = 0; s < StateCount; s++) _stateCounts[s] = 0;

            float maxSpeed = config.maxSpeed;
            float minSpeed = config.minSpeed;
            for (int i = 0; i < _count; i++)
            {
                Vector3 v = _velocities[i] + _accelerations[i] * dt;

                float speed = v.magnitude;
                if (speed < 1e-5f)
                {
                    Vector3 prev = _velocities[i];
                    v = (prev.sqrMagnitude > 1e-10f ? prev.normalized : Vector3.forward) * minSpeed;
                }
                else
                {
                    v = v / speed * Mathf.Clamp(speed, minSpeed, maxSpeed);
                }

                _velocities[i] = v;
                _positions[i] += v * dt;

                int s = (int)_states[i];
                _stateMatrices[s][_stateCounts[s]++] =
                    Matrix4x4.TRS(_positions[i], Quaternion.LookRotation(v, Vector3.up), fishScale);
            }

            _simTimer.Stop();
            _lastSimMs = _simTimer.Elapsed.TotalMilliseconds;

            RenderFish();
        }

        /// <summary>Snapshot predator and foraging-node positions into flat arrays for this frame.</summary>
        void SnapshotThreatsAndNodes()
        {
            var preds = Predator.All;
            _predatorCount = preds.Count;
            if (_predatorPositions.Length < _predatorCount)
                _predatorPositions = new Vector3[Mathf.NextPowerOfTwo(_predatorCount)];
            for (int i = 0; i < _predatorCount; i++)
                _predatorPositions[i] = preds[i].transform.position;

            var nodes = ForagingNode.All;
            _nodeCount = nodes.Count;
            if (_nodePositions.Length < _nodeCount)
                _nodePositions = new Vector3[Mathf.NextPowerOfTwo(_nodeCount)];
            for (int i = 0; i < _nodeCount; i++)
                _nodePositions[i] = nodes[i].transform.position;
        }

        float NearestNeighborSqr(int index)
        {
            Vector3 p = _positions[index];
            float best = float.MaxValue;
            for (int n = 0; n < _neighbors.Count; n++)
            {
                float d2 = (_positions[_neighbors[n]] - p).sqrMagnitude;
                if (d2 < best) best = d2;
            }
            return best;
        }

        Vector3 ForagingPull(Vector3 pos, Vector3 vel)
        {
            int nearest = NearestNodeInRange(pos);
            if (nearest < 0) return Vector3.zero;
            Vector3 desiredDir = _nodePositions[nearest] - pos;
            if (desiredDir.sqrMagnitude < 1e-6f) return Vector3.zero;
            Vector3 desired = desiredDir.normalized * config.maxSpeed;
            return Vector3.ClampMagnitude(desired - vel, config.maxSteerForce) * config.foragingWeight;
        }

        int NearestNodeInRange(Vector3 pos)
        {
            float r2 = config.foragingNodeRadius * config.foragingNodeRadius;
            int nearest = -1;
            float best = r2;
            for (int i = 0; i < _nodeCount; i++)
            {
                float d2 = (_nodePositions[i] - pos).sqrMagnitude;
                if (d2 < best) { best = d2; nearest = i; }
            }
            return nearest;
        }

        void TickBehaviors(float dt)
        {
            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            int bucket = Time.frameCount % buckets;
            float tickDt = dt * buckets; // approx real time between a fish's ticks

            for (int i = bucket; i < _count; i += buckets)
            {
                bool nearNode = NearestNodeInRange(_positions[i]) >= 0;
                float nnDist = Mathf.Sqrt(_nearestNeighborSqr[i]);
                _states[i] = FishBehaviorController.Tick(
                    _states[i],
                    _predatorInRange[i],
                    nnDist,
                    nearNode,
                    Random.value,
                    config.regroupNeighborDistance,
                    config.foragingChancePerTick,
                    config.foragingDuration,
                    tickDt,
                    ref _stateTimer[i]);
            }
        }

        /// <summary>Draw each state's fish in its own colour, batched by the 1023-instance limit.</summary>
        void RenderFish()
        {
            const int kBatch = 1023;
            for (int s = 0; s < StateCount; s++)
            {
                int total = _stateCounts[s];
                Matrix4x4[] mats = _stateMatrices[s];
                RenderParams rp = _stateRenderParams[s];
                for (int start = 0; start < total; start += kBatch)
                {
                    int cnt = Mathf.Min(kBatch, total - start);
                    Graphics.RenderMeshInstanced(rp, _fishMesh, 0, mats, cnt, start);
                }
            }
        }

        void HandleInput()
        {
            Keyboard kb = Keyboard.current;
            if (kb != null && kb[toggleModeKey].wasPressedThisFrame)
                ToggleNeighborMode();
        }

        GUIStyle _statsStyle;

        void OnGUI()
        {
            if (!showStats) return;
            if (_statsStyle == null)
                _statsStyle = new GUIStyle(GUI.skin.label) { fontSize = 16, richText = true };

            float frameMs = Time.smoothDeltaTime * 1000f;
            double restMs = frameMs - _lastSimMs; // GPU + present + editor overhead
            string text =
                $"Fish: <b>{_count}</b>\n" +
                $"Mode: <b>{ActiveModeName}</b>   (press [{toggleModeKey}] to toggle)\n" +
                $"Neighbour query: <b>{_lastQueryMs:F2} ms</b>\n" +
                $"Sim CPU total: <b>{_lastSimMs:F2} ms</b>\n" +
                $"Rest (GPU/present): <b>{restMs:F2} ms</b>\n" +
                $"Frame: <b>{frameMs:F1} ms</b>  ({(frameMs > 0f ? 1000f / frameMs : 0f):F0} fps)";

            GUI.color = Color.black;
            GUI.Label(new Rect(12, 12, 520, 150), text, _statsStyle);
            GUI.color = Color.white;
            GUI.Label(new Rect(11, 11, 520, 150), text, _statsStyle);
        }

        void OnDestroy()
        {
            if (_stateMaterials != null)
                foreach (var m in _stateMaterials)
                    if (m != null) Destroy(m);
        }
    }
}
