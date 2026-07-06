using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using FishSwarm.Flocking;
using FishSwarm.Spatial;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace FishSwarm.Core
{
    /// <summary>
    /// Owns the whole school as struct-of-arrays data (positions/velocities) and steps every
    /// fish in one central loop. This data-oriented layout — rather than one MonoBehaviour per
    /// fish — is what ports cleanly to Burst <c>IJobParallelFor</c> + <c>NativeArray</c> in the
    /// scale-up pass: only the loop body moves into a job.
    ///
    /// Neighbour lookups go through a pluggable <see cref="INeighborProvider"/> so the same live
    /// population can flip between the naive O(n²) scan and the spatial grid at runtime. The cost
    /// of just the neighbour work (grid build + all queries) is timed with a <see cref="Stopwatch"/>
    /// and exposed for the benchmark overlay, isolated from steering and render cost.
    ///
    /// Fish are drawn with GPU instancing (<see cref="Graphics.RenderMeshInstanced"/>) straight
    /// from a per-frame matrix array — no per-fish GameObject or Transform — so thousands of fish
    /// cost only a handful of instanced draw calls.
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
        [Tooltip("Fish material asset (URP/Lit with GPU Instancing enabled). Assigning a real asset " +
                 "is what keeps the instancing shader variant from being stripped out of builds — " +
                 "without it, RenderMeshInstanced silently draws nothing in a standalone player.")]
        public Material fishMaterial;
        [Tooltip("Fallback shader if no material asset is assigned (editor convenience only).")]
        public Shader fishShader;
        public Color fishColor = new Color(0.8f, 0.85f, 0.55f, 1f);
        public Vector3 fishScale = new Vector3(0.25f, 0.25f, 0.7f);

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
        Matrix4x4[] _matrices; // per-fish TRS, rebuilt each frame for instanced rendering
        int _count;

        // Reused per-fish neighbour scratch buffer (no per-frame allocation).
        readonly List<int> _neighbors = new List<int>(256);

        // Pluggable neighbour lookup + isolated timing of just that work.
        NaiveNeighborProvider _naive;
        UniformGridNeighborProvider _grid;
        INeighborProvider _active;
        readonly Stopwatch _queryTimer = new Stopwatch();
        readonly Stopwatch _simTimer = new Stopwatch();
        double _lastQueryMs;
        double _lastSimMs;

        Mesh _fishMesh;
        Material _fishMaterial;
        RenderParams _renderParams;

        public int Count => _count;

        /// <summary>Name of the active neighbour-lookup mode (for the benchmark overlay).</summary>
        public string ActiveModeName => _active != null ? _active.Name : "-";

        /// <summary>Milliseconds spent on neighbour lookup last frame (grid build + all queries).</summary>
        public double LastNeighborQueryMs => _lastQueryMs;

        /// <summary>Milliseconds of CPU simulation last frame (query + steering + integration).</summary>
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

            // Prefer a real material asset: it's what keeps the GPU-instancing shader variant in
            // the build. Clone it so we can tint per fishColor without dirtying the shared asset.
            if (fishMaterial != null)
            {
                _fishMaterial = new Material(fishMaterial);
            }
            else
            {
                Shader shader = fishShader != null ? fishShader : Shader.Find("Universal Render Pipeline/Lit");
                if (shader == null) shader = Shader.Find("Standard");
                if (shader == null)
                    Debug.LogError("[FishManager] No fish material/shader found; fish will be invisible.");
                _fishMaterial = new Material(shader);
            }
            _fishMaterial.color = fishColor;
            _fishMaterial.enableInstancing = true;

            // worldBounds must cover where fish can be, or the instanced batches get frustum-culled
            // as a whole. Grow the ocean volume by a margin to be safe.
            Bounds b = bounds.Bounds;
            _renderParams = new RenderParams(_fishMaterial)
            {
                worldBounds = new Bounds(b.center, b.size + Vector3.one * 10f),
                shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
                receiveShadows = false,
            };
        }

        void Spawn(int n)
        {
            _count = Mathf.Max(0, n);
            _positions = new Vector3[_count];
            _velocities = new Vector3[_count];
            _accelerations = new Vector3[_count];
            _matrices = new Matrix4x4[_count];

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
                _matrices[i] = Matrix4x4.TRS(
                    _positions[i], Quaternion.LookRotation(_velocities[i], Vector3.up), fishScale);
            }

            Random.state = prev;
        }

        void Update()
        {
            if (_count == 0) return;

            HandleInput();

            float dt = Time.deltaTime;
            Bounds volume = bounds.Bounds;

            // Time the whole CPU simulation (neighbour lookup + steering + integration), so the
            // overlay can separate CPU cost from the GPU/present time that makes up the rest of
            // the frame. The render submit below is deliberately outside this timer.
            _simTimer.Restart();

            // Build the neighbour acceleration structure for this frame (timed).
            _queryTimer.Reset();
            _queryTimer.Start();
            _active.Build(_positions, _count, config.perceptionRadius, config.maxNeighbors);
            _queryTimer.Stop();

            // Pass 1 — accumulate accelerations from a snapshot of current positions/velocities,
            // so no fish sees another's half-updated state this frame. Only the neighbour query is
            // timed; the steering maths is deliberately outside the stopwatch so the benchmark
            // isolates lookup cost from everything else.
            for (int i = 0; i < _count; i++)
            {
                _queryTimer.Start();
                _active.Query(i, _neighbors);
                _queryTimer.Stop();

                Vector3 accel = BoidSteering.Compute(i, _positions, _velocities, _neighbors, config);
                accel += BoundarySteering.Compute(_positions[i], _velocities[i], volume, config);
                _accelerations[i] = accel;
            }
            _lastQueryMs = _queryTimer.Elapsed.TotalMilliseconds;

            // Pass 2 — integrate, clamp speed, and rebuild each fish's render matrix.
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
                _matrices[i] = Matrix4x4.TRS(
                    _positions[i], Quaternion.LookRotation(v, Vector3.up), fishScale);
            }

            _simTimer.Stop();
            _lastSimMs = _simTimer.Elapsed.TotalMilliseconds;

            RenderFish();
        }

        /// <summary>
        /// Draw all fish with GPU instancing. RenderMeshInstanced takes up to 1023 instances per
        /// call, so large populations are issued in a few batches straight from the matrix array.
        /// </summary>
        void RenderFish()
        {
            const int kBatch = 1023;
            for (int start = 0; start < _count; start += kBatch)
            {
                int cnt = Mathf.Min(kBatch, _count - start);
                Graphics.RenderMeshInstanced(_renderParams, _fishMesh, 0, _matrices, cnt, start);
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
            if (_fishMaterial != null) Destroy(_fishMaterial);
        }
    }
}
