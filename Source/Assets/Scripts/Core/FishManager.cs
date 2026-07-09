using UnityEngine;
using UnityEngine.InputSystem;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using FishSwarm.Behavior;
using FishSwarm.Predators;
using FishSwarm.Jobs;
using Stopwatch = System.Diagnostics.Stopwatch;
using Random = UnityEngine.Random;

namespace FishSwarm.Core
{
    /// <summary>
    /// Owns the whole school as struct-of-arrays <see cref="NativeArray"/> data and steps it either
    /// as Burst-compiled parallel jobs (Burst ON) or as a plain single-threaded main-thread loop
    /// (Burst OFF) — both call the same <see cref="FishSteering"/> statics, so the only difference
    /// is parallelism + Burst codegen. The naive/grid neighbour toggle works in both modes.
    ///
    /// Fish are drawn with GPU instancing, colour-bucketed by state (a counting sort groups each
    /// state's matrices into a contiguous run) — no per-fish GameObject or Transform.
    /// </summary>
    public class FishManager : MonoBehaviour
    {
        const int StateCount = 4;

        [Header("References (auto-resolved if left empty)")]
        public FishConfig config;
        public OceanBounds bounds;

        [Header("Population")]
        [Min(0)] public int spawnCount = 3000;
        public int randomSeed = 12345;
        [Tooltip("Fish spawn within this radius of the volume centre so they start as one school.")]
        public float spawnRadius = 20f;

        [Header("Fish appearance")]
        [Tooltip("Fish material asset (URP/Lit with GPU Instancing enabled) — keeps the instancing " +
                 "shader variant in standalone builds.")]
        public Material fishMaterial;
        [Tooltip("Fallback shader if no material asset is assigned (editor convenience only).")]
        public Shader fishShader;
        public Vector3 fishScale = new Vector3(0.25f, 0.25f, 0.7f);
        [Tooltip("Colour per state: 0 Schooling, 1 Fleeing, 2 Regrouping, 3 Foraging.")]
        public Color[] stateColors =
        {
            new Color(0.80f, 0.85f, 0.55f),
            new Color(0.90f, 0.20f, 0.15f),
            new Color(0.20f, 0.60f, 0.95f),
            new Color(0.30f, 0.85f, 0.35f),
        };

        [Header("Simulation")]
        [Tooltip("Burst-compiled parallel jobs vs single-threaded main-thread loop (live toggle).")]
        public bool useBurst = true;
        [Tooltip("Spatial grid vs naive O(n²) neighbour lookup (live toggle).")]
        public bool useGrid = true;
        public Key burstToggleKey = Key.B;
        public Key gridToggleKey = Key.Tab;
        public bool showStats = true;

        // Simulation state (NativeArrays, Persistent).
        NativeArray<float3> _positions, _velocities, _accelerations;
        NativeArray<FishState> _states;
        NativeArray<float> _stateTimer, _nearestNeighborSqr;
        NativeArray<byte> _predatorInRange;
        NativeArray<float4x4> _matrices, _sortedMatrices;
        NativeArray<int> _stateStart, _stateCount;
        NativeArray<float3> _predatorPositions, _nodePositions;
        NativeParallelMultiHashMap<int, int> _grid;
        int _count, _predatorCount, _nodeCount;
        bool _allocated;

        // Rendering.
        Mesh _fishMesh;
        Material[] _stateMaterials;
        RenderParams[] _stateRenderParams;

        // Timing.
        readonly Stopwatch _simTimer = new Stopwatch();
        readonly Stopwatch _steerTimer = new Stopwatch();
        double _lastSimMs, _lastSteerMs;
        Vector3 _schoolCenter;

        public int Count => _count;
        public Vector3 SchoolCenter => _schoolCenter;
        public bool UseBurst => useBurst;
        public string ActiveModeName => useGrid ? "Uniform Grid" : "Naive O(n^2)";
        public double LastSimMs => _lastSimMs;
        public double LastNeighborQueryMs => _lastSteerMs;

        void Start()
        {
            if (config == null) config = FishConfig.CreateDefault();
            if (bounds == null) bounds = FindFirstObjectByType<OceanBounds>();
            if (bounds == null)
            {
                Debug.LogError("[FishManager] No OceanBounds found in the scene; cannot spawn.");
                enabled = false;
                return;
            }

            BuildRenderResources();
            Allocate(spawnCount);
            SpawnInit();
        }

        // ---- Allocation ---------------------------------------------------------------------

        void Allocate(int n)
        {
            DisposeArrays();
            _count = Mathf.Max(0, n);
            int cap = Mathf.Max(1, _count);

            _positions = new NativeArray<float3>(cap, Allocator.Persistent);
            _velocities = new NativeArray<float3>(cap, Allocator.Persistent);
            _accelerations = new NativeArray<float3>(cap, Allocator.Persistent);
            _states = new NativeArray<FishState>(cap, Allocator.Persistent);
            _stateTimer = new NativeArray<float>(cap, Allocator.Persistent);
            _nearestNeighborSqr = new NativeArray<float>(cap, Allocator.Persistent);
            _predatorInRange = new NativeArray<byte>(cap, Allocator.Persistent);
            _matrices = new NativeArray<float4x4>(cap, Allocator.Persistent);
            _sortedMatrices = new NativeArray<float4x4>(cap, Allocator.Persistent);
            _stateStart = new NativeArray<int>(StateCount, Allocator.Persistent);
            _stateCount = new NativeArray<int>(StateCount, Allocator.Persistent);
            _predatorPositions = new NativeArray<float3>(8, Allocator.Persistent);
            _nodePositions = new NativeArray<float3>(8, Allocator.Persistent);
            _grid = new NativeParallelMultiHashMap<int, int>(cap, Allocator.Persistent);
            _allocated = true;
        }

        void SpawnInit()
        {
            Random.State prev = Random.state;
            if (randomSeed != 0) Random.InitState(randomSeed);

            float startSpeed = 0.5f * (config.minSpeed + config.maxSpeed);
            Vector3 centre = bounds.Bounds.center;
            for (int i = 0; i < _count; i++)
            {
                Vector3 pos = spawnRadius > 0f
                    ? bounds.Clamp(centre + Random.insideUnitSphere * spawnRadius)
                    : bounds.RandomPointInside();
                _positions[i] = pos;
                _velocities[i] = (float3)(Random.onUnitSphere * startSpeed);
                _states[i] = FishState.Schooling;
                _stateTimer[i] = 0f;
                _matrices[i] = float4x4.TRS(pos, quaternion.LookRotationSafe(_velocities[i], math.up()), fishScale);
            }

            Random.state = prev;
            UpdateSchoolCenter();
        }

        void EnsureFloat3Capacity(ref NativeArray<float3> arr, int n)
        {
            if (arr.IsCreated && arr.Length >= Mathf.Max(1, n)) return;
            if (arr.IsCreated) arr.Dispose();
            arr = new NativeArray<float3>(Mathf.Max(1, n), Allocator.Persistent);
        }

        void EnsureGridCapacity(int n)
        {
            if (_grid.IsCreated && _grid.Capacity >= n) return;
            if (_grid.IsCreated) _grid.Dispose();
            _grid = new NativeParallelMultiHashMap<int, int>(Mathf.Max(1, n), Allocator.Persistent);
        }

        void DisposeArrays()
        {
            if (!_allocated) return;
            _positions.Dispose(); _velocities.Dispose(); _accelerations.Dispose();
            _states.Dispose(); _stateTimer.Dispose(); _nearestNeighborSqr.Dispose();
            _predatorInRange.Dispose(); _matrices.Dispose(); _sortedMatrices.Dispose();
            _stateStart.Dispose(); _stateCount.Dispose();
            _predatorPositions.Dispose(); _nodePositions.Dispose();
            if (_grid.IsCreated) _grid.Dispose();
            _allocated = false;
        }

        // ---- Rendering resources ------------------------------------------------------------

        void BuildRenderResources()
        {
            var temp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _fishMesh = temp.GetComponent<MeshFilter>().sharedMesh;
            Destroy(temp);

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
                var m = new Material(baseMat) { color = (stateColors != null && s < stateColors.Length) ? stateColors[s] : Color.white };
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

        // ---- Update -------------------------------------------------------------------------

        void Update()
        {
            if (_count == 0) return;

            HandleInput();
            SnapshotThreatsAndNodes();

            var p = SteeringParams.From(config, bounds.Bounds.center, bounds.Bounds.size);
            float dt = Time.deltaTime;

            _simTimer.Restart();
            if (useBurst) RunBurst(p, dt);
            else RunManaged(p, dt);
            _simTimer.Stop();
            _lastSimMs = _simTimer.Elapsed.TotalMilliseconds;

            UpdateSchoolCenter();
            RenderFish();
        }

        void RunBurst(SteeringParams p, float dt)
        {
            JobHandle gridH = default;
            if (useGrid)
            {
                EnsureGridCapacity(_count);
                _grid.Clear();
                gridH = new BuildGridJob
                {
                    positions = _positions,
                    p = p,
                    grid = _grid.AsParallelWriter(),
                }.Schedule(_count, 64);
            }

            _steerTimer.Restart();
            var steerH = new SteeringJob
            {
                positions = _positions, velocities = _velocities, states = _states,
                predators = _predatorPositions, predatorCount = _predatorCount,
                nodes = _nodePositions, nodeCount = _nodeCount,
                grid = _grid, useGrid = useGrid, p = p,
                accelerations = _accelerations, nearestNeighborSqr = _nearestNeighborSqr,
                predatorInRange = _predatorInRange,
            }.Schedule(_count, 64, gridH);
            steerH.Complete();
            _steerTimer.Stop();
            _lastSteerMs = _steerTimer.Elapsed.TotalMilliseconds;

            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            var behaviorH = new BehaviorJob
            {
                positions = _positions, nearestNeighborSqr = _nearestNeighborSqr,
                predatorInRange = _predatorInRange, nodes = _nodePositions, nodeCount = _nodeCount,
                states = _states, stateTimer = _stateTimer, p = p,
                frameBucket = Time.frameCount % buckets, buckets = buckets,
                tickDt = dt * buckets, frameSeed = (uint)(Time.frameCount * 9781 + 1),
            }.Schedule(_count, 64);

            // Integrate must run after behaviour (behaviour reads positions; integrate writes them).
            var integrateH = new IntegrateJob
            {
                positions = _positions, velocities = _velocities, accelerations = _accelerations,
                matrices = _matrices, dt = dt, p = p, scale = fishScale,
            }.Schedule(_count, 64, behaviorH);

            new BucketByStateJob
            {
                count = _count, states = _states, matrices = _matrices,
                sorted = _sortedMatrices, stateStart = _stateStart, stateCount = _stateCount,
            }.Schedule(integrateH).Complete();
        }

        void RunManaged(SteeringParams p, float dt)
        {
            if (useGrid)
            {
                EnsureGridCapacity(_count);
                _grid.Clear();
                for (int i = 0; i < _count; i++)
                {
                    int3 c = FishSteering.CellCoords(_positions[i], p);
                    _grid.Add(FishSteering.CellKey(c, p), i);
                }
            }

            _steerTimer.Restart();
            for (int i = 0; i < _count; i++)
            {
                FishSteering.Step(i, _positions, _velocities, _states, _predatorPositions, _predatorCount,
                    _nodePositions, _nodeCount, _grid, useGrid, p, _accelerations, _nearestNeighborSqr, _predatorInRange);
            }
            _steerTimer.Stop();
            _lastSteerMs = _steerTimer.Elapsed.TotalMilliseconds;

            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            int frameBucket = Time.frameCount % buckets;
            float tickDt = dt * buckets;
            uint seed = (uint)(Time.frameCount * 9781 + 1);
            for (int i = frameBucket; i < _count; i += buckets)
            {
                FishSteering.TickBehavior(i, _positions, _nearestNeighborSqr, _predatorInRange,
                    _nodePositions, _nodeCount, _states, _stateTimer, p, tickDt, seed);
            }

            for (int i = 0; i < _count; i++)
            {
                FishSteering.Integrate(i, _positions, _velocities, _accelerations, _matrices, dt, p, fishScale);
            }

            FishSteering.BucketByState(_count, _states, _matrices, _sortedMatrices, _stateStart, _stateCount);
        }

        void SnapshotThreatsAndNodes()
        {
            var preds = Predator.All;
            _predatorCount = preds.Count;
            EnsureFloat3Capacity(ref _predatorPositions, _predatorCount);
            for (int i = 0; i < _predatorCount; i++)
                _predatorPositions[i] = preds[i].transform.position;

            var nodes = ForagingNode.All;
            _nodeCount = nodes.Count;
            EnsureFloat3Capacity(ref _nodePositions, _nodeCount);
            for (int i = 0; i < _nodeCount; i++)
                _nodePositions[i] = nodes[i].transform.position;
        }

        void RenderFish()
        {
            const int kBatch = 1023;
            var mats = _sortedMatrices.Reinterpret<Matrix4x4>();
            for (int s = 0; s < StateCount; s++)
            {
                int start = _stateStart[s];
                int total = _stateCount[s];
                for (int off = 0; off < total; off += kBatch)
                {
                    int cnt = Mathf.Min(kBatch, total - off);
                    var sub = mats.GetSubArray(start + off, cnt);
                    Graphics.RenderMeshInstanced(_stateRenderParams[s], _fishMesh, 0, sub);
                }
            }
        }

        void UpdateSchoolCenter()
        {
            if (_count == 0)
            {
                _schoolCenter = bounds != null ? bounds.Bounds.center : Vector3.zero;
                return;
            }

            const int maxSamples = 256;
            int step = Mathf.Max(1, _count / maxSamples);
            int samples = 0;
            Vector3 sum = Vector3.zero;

            for (int i = 0; i < _count; i += step)
            {
                sum += (Vector3)_positions[i];
                samples++;
            }

            _schoolCenter = samples > 0 ? sum / samples : (bounds != null ? bounds.Bounds.center : Vector3.zero);
        }

        void HandleInput()
        {
            Keyboard kb = Keyboard.current;
            if (kb == null) return;
            if (kb[gridToggleKey].wasPressedThisFrame) useGrid = !useGrid;
            if (kb[burstToggleKey].wasPressedThisFrame) useBurst = !useBurst;
        }

        GUIStyle _statsStyle;

        void OnGUI()
        {
            if (!showStats) return;
            if (_statsStyle == null)
                _statsStyle = new GUIStyle(GUI.skin.label) { fontSize = 16, richText = true };

            float frameMs = Time.smoothDeltaTime * 1000f;
            double restMs = frameMs - _lastSimMs;
            string text =
                $"Fish: <b>{_count}</b>\n" +
                $"Burst: <b>{(useBurst ? "ON" : "OFF")}</b>   (press [{burstToggleKey}])\n" +
                $"Mode: <b>{ActiveModeName}</b>   (press [{gridToggleKey}])\n" +
                $"Neighbour+steer: <b>{_lastSteerMs:F2} ms</b>\n" +
                $"Sim CPU total: <b>{_lastSimMs:F2} ms</b>\n" +
                $"Rest (GPU/present): <b>{restMs:F2} ms</b>\n" +
                $"Frame: <b>{frameMs:F1} ms</b>  ({(frameMs > 0f ? 1000f / frameMs : 0f):F0} fps)";

            GUI.color = Color.black;
            GUI.Label(new Rect(12, 12, 560, 170), text, _statsStyle);
            GUI.color = Color.white;
            GUI.Label(new Rect(11, 11, 560, 170), text, _statsStyle);
        }

        void OnDestroy()
        {
            DisposeArrays();
            if (_stateMaterials != null)
                foreach (var m in _stateMaterials)
                    if (m != null) Destroy(m);
        }
    }
}
