using System.Collections.Generic;
using UnityEngine;

namespace FishSwarm.Predators
{
    /// <summary>
    /// Minimal predator: a moving position the fish flee from. It carries a simple placeholder
    /// ping-pong patrol so the split/reform is testable now; Step 4 replaces the movement with a
    /// proper state machine (waypoint/spline patrol + occasional lunge). Self-registers so the
    /// FishManager can read every predator's position each frame.
    /// </summary>
    public class Predator : MonoBehaviour
    {
        public static readonly List<Predator> All = new List<Predator>();

        [Tooltip("Radius shown as a gizmo — roughly where fish begin to flee (tune the actual " +
                 "flee distance on FishConfig.predatorFleeRadius).")]
        public float threatRadius = 14f;

        [Header("Placeholder patrol (Step 4 replaces this)")]
        public bool autoPatrol = true;
        [Tooltip("Half-length of the ping-pong sweep, measured from the start position.")]
        public float patrolDistance = 55f;
        public Vector3 patrolAxis = Vector3.right;
        public float speed = 12f;

        Vector3 _start;

        void OnEnable()
        {
            All.Add(this);
            _start = transform.position;
        }

        void OnDisable()
        {
            All.Remove(this);
        }

        void Update()
        {
            if (!autoPatrol) return;
            Vector3 axis = patrolAxis.sqrMagnitude > 1e-6f ? patrolAxis.normalized : Vector3.right;
            float offset = Mathf.PingPong(Time.time * speed, patrolDistance * 2f) - patrolDistance;
            transform.position = _start + axis * offset;
        }

        void OnDrawGizmos()
        {
            Gizmos.color = new Color(1f, 0.2f, 0.15f, 0.5f);
            Gizmos.DrawWireSphere(transform.position, threatRadius);
        }
    }
}
