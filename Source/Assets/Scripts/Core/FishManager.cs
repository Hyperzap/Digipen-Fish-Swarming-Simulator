using System.Collections.Generic;
using UnityEngine;
using FishSwarm.Flocking;

namespace FishSwarm.Core
{
    /// <summary>
    /// Owns the whole school as struct-of-arrays data (positions/velocities) and steps every
    /// fish in one central loop. This data-oriented layout — rather than one MonoBehaviour per
    /// fish — is what ports cleanly to Burst <c>IJobParallelFor</c> + <c>NativeArray</c> in the
    /// scale-up pass: only the loop body moves into a job.
    ///
    /// Step 1 uses a naive O(n²) neighbour scan (<see cref="GatherNeighborsNaive"/>). Step 2
    /// swaps that single call for a pluggable spatial-grid provider without touching steering.
    /// Rendering is a pooled Transform per fish; it becomes GPU instancing at scale.
    /// </summary>
    public class FishManager : MonoBehaviour
    {
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
        public Color fishColor = new Color(0.8f, 0.85f, 0.55f, 1f);
        public Vector3 fishScale = new Vector3(0.25f, 0.25f, 0.7f);

        // Struct-of-arrays simulation state.
        Vector3[] _positions;
        Vector3[] _velocities;
        Vector3[] _accelerations;
        Transform[] _transforms;
        int _count;

        // Reused per-fish neighbour scratch buffer (no per-frame allocation).
        readonly List<int> _neighbors = new List<int>(256);

        Mesh _fishMesh;
        Material _fishMaterial;
        Transform _fishRoot;

        public int Count => _count;

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

            BuildRenderResources();
            Spawn(spawnCount);
        }

        void BuildRenderResources()
        {
            // Borrow a built-in cube mesh, then discard the temporary primitive (and its collider).
            var temp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _fishMesh = temp.GetComponent<MeshFilter>().sharedMesh;
            Destroy(temp);

            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            _fishMaterial = new Material(shader) { color = fishColor };

            _fishRoot = new GameObject("Fish").transform;
            _fishRoot.SetParent(transform, false);
        }

        void Spawn(int n)
        {
            _count = Mathf.Max(0, n);
            _positions = new Vector3[_count];
            _velocities = new Vector3[_count];
            _accelerations = new Vector3[_count];
            _transforms = new Transform[_count];

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

                var go = new GameObject("Fish_" + i);
                go.transform.SetParent(_fishRoot, false);
                var mf = go.AddComponent<MeshFilter>();
                mf.sharedMesh = _fishMesh;
                var mr = go.AddComponent<MeshRenderer>();
                mr.sharedMaterial = _fishMaterial;
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                go.transform.localScale = fishScale;
                _transforms[i] = go.transform;
            }

            Random.state = prev;
        }

        void Update()
        {
            if (_count == 0) return;
            float dt = Time.deltaTime;
            Bounds volume = bounds.Bounds;

            // Pass 1 — accumulate accelerations from a snapshot of current positions/velocities,
            // so no fish sees another's half-updated state this frame.
            for (int i = 0; i < _count; i++)
            {
                GatherNeighborsNaive(i, _neighbors);
                Vector3 accel = BoidSteering.Compute(i, _positions, _velocities, _neighbors, config);
                accel += BoundarySteering.Compute(_positions[i], _velocities[i], volume, config);
                _accelerations[i] = accel;
            }

            // Pass 2 — integrate, clamp speed, move + orient the render transforms.
            float maxSpeed = config.maxSpeed;
            float minSpeed = config.minSpeed;
            for (int i = 0; i < _count; i++)
            {
                Vector3 v = _velocities[i] + _accelerations[i] * dt;

                float speed = v.magnitude;
                if (speed < 1e-5f)
                    v = _transforms[i].forward * minSpeed;
                else
                    v = v / speed * Mathf.Clamp(speed, minSpeed, maxSpeed);

                _velocities[i] = v;
                _positions[i] += v * dt;
                _transforms[i].SetPositionAndRotation(
                    _positions[i], Quaternion.LookRotation(v, Vector3.up));
            }
        }

        /// <summary>
        /// Step-1 baseline neighbour query: brute-force all-pairs distance test. Fills
        /// <paramref name="result"/> with indices within the perception radius of fish
        /// <paramref name="index"/>. Step 2 replaces this with a spatial-grid lookup.
        /// </summary>
        void GatherNeighborsNaive(int index, List<int> result)
        {
            result.Clear();
            Vector3 p = _positions[index];
            float r2 = config.perceptionRadius * config.perceptionRadius;
            for (int j = 0; j < _count; j++)
            {
                if (j == index) continue;
                if ((_positions[j] - p).sqrMagnitude < r2)
                    result.Add(j);
            }
        }

        void OnDestroy()
        {
            if (_fishMaterial != null) Destroy(_fishMaterial);
        }
    }
}
