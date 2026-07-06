using System.Collections.Generic;
using UnityEngine;
using FishSwarm.Core;

namespace FishSwarm.Flocking
{
    /// <summary>
    /// Classic Reynolds boid steering — separation, alignment, cohesion — computed over a
    /// pre-gathered list of neighbour indices. Deliberately stateless so it can be reused
    /// unchanged when the neighbour list comes from the naive O(n²) scan (step 1) or from the
    /// spatial grid (step 2). Threat-avoidance is layered on top in a later step.
    /// </summary>
    public static class BoidSteering
    {
        /// <summary>
        /// Accumulated steering acceleration for fish <paramref name="index"/> given its
        /// neighbours. Reads current positions/velocities only — the caller integrates.
        /// </summary>
        /// <param name="sacMul">Per-state multipliers for (separation, alignment, cohesion), so the
        /// behaviour layer can, for example, drop cohesion while fleeing without touching the base
        /// weights on the config.</param>
        public static Vector3 Compute(
            int index,
            Vector3[] positions,
            Vector3[] velocities,
            List<int> neighbors,
            FishConfig cfg,
            Vector3 sacMul)
        {
            Vector3 pos = positions[index];
            Vector3 vel = velocities[index];

            Vector3 separation = Vector3.zero;
            Vector3 alignmentSum = Vector3.zero;
            Vector3 cohesionSum = Vector3.zero;
            int flockCount = 0;

            float sepSqr = cfg.separationRadius * cfg.separationRadius;

            for (int n = 0; n < neighbors.Count; n++)
            {
                int j = neighbors[n];
                Vector3 offset = pos - positions[j];
                float d2 = offset.sqrMagnitude;
                if (d2 < 1e-6f)
                    continue; // coincident / self

                // Separation: only within personal space, weighted stronger the closer it is.
                if (d2 < sepSqr)
                    separation += offset / d2;

                // Alignment + cohesion use the full perception set (already radius-filtered by caller).
                alignmentSum += velocities[j];
                cohesionSum += positions[j];
                flockCount++;
            }

            Vector3 force = Vector3.zero;

            if (flockCount > 0)
            {
                Vector3 avgVelocity = alignmentSum / flockCount;
                force += Steer(avgVelocity, vel, cfg) * (cfg.alignmentWeight * sacMul.y);

                // Anisotropic cohesion. We steer toward the centroid the usual (Reynolds) way,
                // then scale the resulting force PER AXIS. Scaling the force *magnitude* — rather
                // than only biasing a direction that Steer() would re-normalise back to full
                // strength — makes the restoring stiffness genuinely different per axis. The
                // school's equilibrium extent then runs roughly proportional to 1 / axisWeight,
                // giving a clearly ellipsoidal school (e.g. weight 3 on Y => ~1/3 the vertical
                // extent) instead of a sphere.
                Vector3 toCentre = (cohesionSum / flockCount) - pos;
                Vector3 cohesionForce = Steer(toCentre, vel, cfg);
                cohesionForce.Scale(cfg.cohesionAxisWeights);
                force += cohesionForce * (cfg.cohesionWeight * sacMul.z);
            }

            if (separation.sqrMagnitude > 1e-8f)
                force += Steer(separation, vel, cfg) * (cfg.separationWeight * sacMul.x);

            return force;
        }

        /// <summary>
        /// Reynolds steering: desired velocity (full speed toward the given direction) minus
        /// current velocity, clamped to the maximum steering force.
        /// </summary>
        static Vector3 Steer(Vector3 desiredDir, Vector3 velocity, FishConfig cfg)
        {
            if (desiredDir.sqrMagnitude < 1e-8f)
                return Vector3.zero;
            Vector3 desired = desiredDir.normalized * cfg.maxSpeed;
            return Vector3.ClampMagnitude(desired - velocity, cfg.maxSteerForce);
        }
    }
}
