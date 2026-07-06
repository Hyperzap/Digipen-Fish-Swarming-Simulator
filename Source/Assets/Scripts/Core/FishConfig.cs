using UnityEngine;

namespace FishSwarm.Core
{
    /// <summary>
    /// Tunable steering parameters shared by every fish. Lives as a ScriptableObject asset so
    /// the flocking feel can be dialled in from one place. Behaviour states (Fleeing, etc.)
    /// added later will scale these weights rather than replace them.
    /// </summary>
    [CreateAssetMenu(fileName = "FishConfig", menuName = "Fish Swarm/Fish Config")]
    public class FishConfig : ScriptableObject
    {
        [Header("Perception")]
        [Tooltip("Neighbour query radius for alignment and cohesion.")]
        public float perceptionRadius = 4f;
        [Tooltip("Closer-range personal-space radius for separation (<= perception).")]
        public float separationRadius = 1.5f;

        [Header("Steering weights")]
        public float separationWeight = 1.6f;
        public float alignmentWeight = 1.0f;
        public float cohesionWeight = 1.0f;
        [Tooltip("How hard fish turn away from the volume edges.")]
        public float boundaryWeight = 2.0f;

        [Header("School shape")]
        [Tooltip("Per-axis cohesion stiffness. Equilibrium extent runs roughly as 1/weight, so a " +
                 "non-uniform vector makes the school ellipsoidal instead of spherical. " +
                 "e.g. (1, 3, 0.6) => vertical extent ~1/3 of width, and ~1.6x longer along Z.")]
        public Vector3 cohesionAxisWeights = new Vector3(1f, 3f, 0.6f);

        [Header("Movement")]
        public float maxSpeed = 6f;
        [Tooltip("Fish never drop below this speed, so the school always looks alive.")]
        public float minSpeed = 2f;
        [Tooltip("Maximum magnitude of any single steering force (Reynolds clamp).")]
        public float maxSteerForce = 8f;

        [Header("Boundary")]
        [Tooltip("Distance from a volume face at which fish begin steering back inward.")]
        public float boundaryMargin = 6f;

        /// <summary>A runtime default instance used when no config asset is wired up.</summary>
        public static FishConfig CreateDefault() => CreateInstance<FishConfig>();
    }
}
