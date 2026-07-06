using UnityEngine;
using FishSwarm.Core;

namespace FishSwarm.Flocking
{
    /// <summary>
    /// Strong repulsion from nearby predators, scaled by <see cref="FishConfig.threatWeight"/> so
    /// it dominates separation/alignment/cohesion when a predator is close. Fish nearest the
    /// predator get the biggest push while distant fish barely react — the school splitting around
    /// the predator is emergent from that per-fish difference, never explicit group logic.
    /// </summary>
    public static class ThreatSteering
    {
        /// <param name="predatorInRange">True if any predator is within the flee radius (used to
        /// trigger the Fleeing state).</param>
        public static Vector3 Compute(
            Vector3 pos, Vector3 vel, Vector3[] predators, int predatorCount,
            FishConfig cfg, out bool predatorInRange)
        {
            predatorInRange = false;
            Vector3 flee = Vector3.zero;
            float r = cfg.predatorFleeRadius;
            float r2 = r * r;

            for (int p = 0; p < predatorCount; p++)
            {
                Vector3 off = pos - predators[p];
                float d2 = off.sqrMagnitude;
                if (d2 < r2 && d2 > 1e-6f)
                {
                    predatorInRange = true;
                    float d = Mathf.Sqrt(d2);
                    flee += (off / d) * (1f - d / r); // away from predator, ramps up as it nears
                }
            }

            if (flee.sqrMagnitude < 1e-8f)
                return Vector3.zero;

            Vector3 desired = flee.normalized * cfg.maxSpeed;
            return Vector3.ClampMagnitude(desired - vel, cfg.maxSteerForce) * cfg.threatWeight;
        }
    }
}
