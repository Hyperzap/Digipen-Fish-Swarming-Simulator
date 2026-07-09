using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using FishSwarm.Core;
using FishSwarm.FSM;

namespace FishSwarm.Predators
{
    /// <summary>
    /// Predator controller with its own FSM. It targets the sampled school centre rather than
    /// individual fish, which keeps predator logic cheap for the instanced fish simulation.
    /// </summary>
    public class Predator : MonoBehaviour
    {
        public static readonly List<Predator> All = new List<Predator>();

        [Header("State")]
        public PredatorState state = PredatorState.Patrol;
        [Tooltip("Fish manager used as the prey target source. Auto-resolved if left empty.")]
        public FishManager fishManager;

        [Header("Sensing")]
        [Tooltip("Radius shown as a gizmo and used as the predator FSM detection range.")]
        public float threatRadius = 14f;
        [Tooltip("Distance at which Chase becomes Lunge.")]
        public float lungeRange = 8f;

        [Header("Timing")]
        public float lungeDuration = 0.8f;
        public float recoverDuration = 1.2f;

        [Header("Movement")]
        [FormerlySerializedAs("speed")]
        public float patrolSpeed = 12f;
        public float chaseSpeed = 18f;
        public float lungeSpeed = 32f;
        public float recoverSpeed = 6f;
        [Tooltip("How quickly the predator turns toward its desired heading.")]
        public float turnSharpness = 8f;

        [Header("Patrol")]
        [Tooltip("If enabled, Patrol follows the ping-pong sweep. If disabled, Patrol idles.")]
        public bool autoPatrol = true;
        [Tooltip("Half-length of the ping-pong sweep, measured from the start position.")]
        public float patrolDistance = 55f;
        public Vector3 patrolAxis = Vector3.right;

        Vector3 _start;
        Vector3 _heading = Vector3.right;
        Vector3 _lungeHeading = Vector3.right;
        float _stateTimer;

        void OnEnable()
        {
            All.Add(this);
            _start = transform.position;
            if (fishManager == null) fishManager = FindFirstObjectByType<FishManager>();
        }

        void OnDisable()
        {
            All.Remove(this);
        }

        void Update()
        {
            if (fishManager == null) fishManager = FindFirstObjectByType<FishManager>();

            Vector3 preyTarget = fishManager != null ? fishManager.SchoolCenter : transform.position;
            var input = new PredatorStateInputs
            {
                HasPreyTarget = fishManager != null && fishManager.Count > 0,
                DistanceToPrey = Vector3.Distance(transform.position, preyTarget),
            };

            var settings = new PredatorStateSettings
            {
                DetectionRadius = threatRadius,
                LungeRange = lungeRange,
                LungeDuration = lungeDuration,
                RecoverDuration = recoverDuration,
            };

            PredatorState previous = state;
            state = PredatorStateMachine.Tick(state, input, settings, Time.deltaTime, ref _stateTimer);
            if (previous != PredatorState.Lunge && state == PredatorState.Lunge)
                _lungeHeading = DesiredHeadingTo(preyTarget);

            MoveForState(preyTarget);
        }

        void MoveForState(Vector3 preyTarget)
        {
            switch (state)
            {
                case PredatorState.Chase:
                    MoveAlong(DesiredHeadingTo(preyTarget), chaseSpeed);
                    break;

                case PredatorState.Lunge:
                    MoveAlong(_lungeHeading, lungeSpeed);
                    break;

                case PredatorState.Recover:
                    MoveAlong(_heading, recoverSpeed);
                    break;

                case PredatorState.Patrol:
                default:
                    UpdatePatrol();
                    break;
            }
        }

        void UpdatePatrol()
        {
            if (!autoPatrol) return;

            Vector3 axis = patrolAxis.sqrMagnitude > 1e-6f ? patrolAxis.normalized : Vector3.right;
            float offset = Mathf.PingPong(Time.time * patrolSpeed, patrolDistance * 2f) - patrolDistance;
            Vector3 target = _start + axis * offset;
            MoveTowardPosition(target, patrolSpeed);
        }

        void MoveTowardPosition(Vector3 target, float speed)
        {
            Vector3 toTarget = target - transform.position;
            if (toTarget.sqrMagnitude < 1e-6f) return;
            MoveAlong(toTarget.normalized, speed);
        }

        void MoveAlong(Vector3 desiredHeading, float speed)
        {
            if (desiredHeading.sqrMagnitude < 1e-6f) return;

            float t = 1f - Mathf.Exp(-turnSharpness * Time.deltaTime);
            _heading = Vector3.Slerp(_heading, desiredHeading.normalized, t).normalized;
            transform.position += _heading * speed * Time.deltaTime;
            transform.rotation = Quaternion.LookRotation(_heading, Vector3.up);
        }

        Vector3 DesiredHeadingTo(Vector3 target)
        {
            Vector3 toTarget = target - transform.position;
            return toTarget.sqrMagnitude > 1e-6f ? toTarget.normalized : _heading;
        }

        void OnDrawGizmos()
        {
            Gizmos.color = new Color(1f, 0.2f, 0.15f, 0.5f);
            Gizmos.DrawWireSphere(transform.position, threatRadius);

            Gizmos.color = new Color(1f, 0.8f, 0.1f, 0.5f);
            Gizmos.DrawWireSphere(transform.position, lungeRange);
        }
    }
}
