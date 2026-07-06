using UnityEngine;
using FishSwarm.Core;

namespace FishSwarm.Flocking
{
    /// <summary>
    /// Soft steer-back force that keeps fish inside the ocean volume. No hard wrapping or
    /// clamping of position — fish smoothly turn inward as they approach a face, so the school
    /// stays coherent near edges instead of teleporting or piling up.
    /// </summary>
    public static class BoundarySteering
    {
        public static Vector3 Compute(Vector3 pos, Vector3 velocity, Bounds bounds, FishConfig cfg)
        {
            float margin = cfg.boundaryMargin;
            Vector3 min = bounds.min;
            Vector3 max = bounds.max;

            Vector3 inward = Vector3.zero;
            if (pos.x < min.x + margin) inward.x += 1f;
            else if (pos.x > max.x - margin) inward.x -= 1f;
            if (pos.y < min.y + margin) inward.y += 1f;
            else if (pos.y > max.y - margin) inward.y -= 1f;
            if (pos.z < min.z + margin) inward.z += 1f;
            else if (pos.z > max.z - margin) inward.z -= 1f;

            if (inward.sqrMagnitude < 1e-8f)
                return Vector3.zero;

            Vector3 desired = inward.normalized * cfg.maxSpeed;
            return Vector3.ClampMagnitude(desired - velocity, cfg.maxSteerForce) * cfg.boundaryWeight;
        }
    }
}
