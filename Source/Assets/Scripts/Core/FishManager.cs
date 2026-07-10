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

        // Cell-sorted grid: ping-pong buffers (data reordered into cell order each frame) plus the
        // counting-sort scratch/offsets. Steering reads the sorted B buffers for contiguous access,
        // then the pair is swapped so the sorted result becomes canonical for next frame.
        NativeArray<float3> _posB, _velB;
        NativeArray<FishState> _stateB;
        NativeArray<float> _timerB;
        NativeArray<int> _cellIndex, _cellCounts, _cellStart;

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

        int _pendingCount = -1;

        public int Count => _count;
        public Vector3 SchoolCenter => _schoolCenter;
        public bool UseBurst => useBurst;
        public bool UseGrid => useGrid;
        public string ActiveModeName => useGrid ? "Uniform Grid" : "Naive O(n^2)";
        public double LastSimMs => _lastSimMs;
        public double LastNeighborQueryMs => _lastSteerMs;
        public FishConfig Config => config;

        /// <summary>Request a new population; applied at the top of the next Update (race-free).</summary>
        public void RequestPopulation(int n) => _pendingCount = Mathf.Max(0, n);

        /// <summary>Toggle helpers so UI buttons can mirror the B / Tab keys.</summary>
        public void SetBurst(bool on) => useBurst = on;
        public void SetGrid(bool on) => useGrid = on;

        bool _ownsConfig;

        void Awake()
        {
            // Work on a runtime copy of the config so live tuning (HUD sliders) never mutates the
            // FishConfig asset on disk. The original asset is the untouched backup; the clone is
            // what's read (SteeringParams snapshot) and written (HUD) every frame, and is discarded
            // on teardown. Done in Awake so it exists before any other component's Start reads it.
            if (config == null) config = FishConfig.CreateDefault();
            else config = Instantiate(config);
            _ownsConfig = true;
        }

        void Start()
        {
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

            _posB = new NativeArray<float3>(cap, Allocator.Persistent);
            _velB = new NativeArray<float3>(cap, Allocator.Persistent);
            _stateB = new NativeArray<FishState>(cap, Allocator.Persistent);
            _timerB = new NativeArray<float>(cap, Allocator.Persistent);
            _cellIndex = new NativeArray<int>(cap, Allocator.Persistent);

            var p0 = SteeringParams.From(config, bounds.Bounds.center, bounds.Bounds.size);
            int cellCount = math.max(1, p0.gridDim.x * p0.gridDim.y * p0.gridDim.z);
            _cellCounts = new NativeArray<int>(cellCount, Allocator.Persistent);
            _cellStart = new NativeArray<int>(cellCount + 1, Allocator.Persistent);
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

        void EnsureGridArrays(int cellCount)
        {
            if (_cellStart.IsCreated && _cellStart.Length >= cellCount + 1) return;
            if (_cellStart.IsCreated) _cellStart.Dispose();
            if (_cellCounts.IsCreated) _cellCounts.Dispose();
            _cellCounts = new NativeArray<int>(cellCount, Allocator.Persistent);
            _cellStart = new NativeArray<int>(cellCount + 1, Allocator.Persistent);
        }

        void DisposeArrays()
        {
            if (!_allocated) return;
            _positions.Dispose(); _velocities.Dispose(); _accelerations.Dispose();
            _states.Dispose(); _stateTimer.Dispose(); _nearestNeighborSqr.Dispose();
            _predatorInRange.Dispose(); _matrices.Dispose(); _sortedMatrices.Dispose();
            _stateStart.Dispose(); _stateCount.Dispose();
            _predatorPositions.Dispose(); _nodePositions.Dispose();
            _posB.Dispose(); _velB.Dispose(); _stateB.Dispose(); _timerB.Dispose();
            _cellIndex.Dispose();
            if (_cellCounts.IsCreated) _cellCounts.Dispose();
            if (_cellStart.IsCreated) _cellStart.Dispose();
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
            // Apply a pending population change here — the previous frame's jobs have completed, so
            // disposing/reallocating the NativeArrays is race-free.
            if (_pendingCount >= 0)
            {
                if (_pendingCount != _count)
                {
                    Allocate(_pendingCount);
                    SpawnInit();
                }
                _pendingCount = -1;
            }

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
            int cellCount = math.max(1, p.gridDim.x * p.gridDim.y * p.gridDim.z);
            EnsureGridArrays(cellCount);

            _steerTimer.Restart();

            // Cell index (parallel) -> counting sort reorders A into cell-order B (single thread).
            var indexH = new BuildCellIndexJob
            {
                positions = _positions, p = p, cellIndex = _cellIndex,
            }.Schedule(_count, 64);

            var sortH = new CountingSortJob
            {
                count = _count, cellCount = cellCount, cellIndex = _cellIndex,
                cellCounts = _cellCounts, cellStart = _cellStart,
                srcPos = _positions, srcVel = _velocities, srcState = _states, srcTimer = _stateTimer,
                dstPos = _posB, dstVel = _velB, dstState = _stateB, dstTimer = _timerB,
            }.Schedule(indexH);

            // Steering over the cell-sorted buffers — contiguous neighbour reads.
            var steerH = new SteeringJob
            {
                positions = _posB, velocities = _velB, states = _stateB,
                predators = _predatorPositions, predatorCount = _predatorCount,
                nodes = _nodePositions, nodeCount = _nodeCount,
                cellStart = _cellStart, useGrid = useGrid, p = p,
                accelerations = _accelerations, nearestNeighborSqr = _nearestNeighborSqr,
                predatorInRange = _predatorInRange,
            }.Schedule(_count, 64, sortH);
            steerH.Complete();
            _steerTimer.Stop();
            _lastSteerMs = _steerTimer.Elapsed.TotalMilliseconds;

            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            var behaviorH = new BehaviorJob
            {
                positions = _posB, nearestNeighborSqr = _nearestNeighborSqr,
                predatorInRange = _predatorInRange, nodes = _nodePositions, nodeCount = _nodeCount,
                states = _stateB, stateTimer = _timerB, p = p,
                frameBucket = Time.frameCount % buckets, buckets = buckets,
                tickDt = dt * buckets, frameSeed = (uint)(Time.frameCount * 9781 + 1),
            }.Schedule(_count, 64);

            // Integrate must run after behaviour (behaviour reads posB; integrate writes it).
            var integrateH = new IntegrateJob
            {
                positions = _posB, velocities = _velB, accelerations = _accelerations,
                matrices = _matrices, dt = dt, p = p, scale = fishScale,
            }.Schedule(_count, 64, behaviorH);

            new BucketByStateJob
            {
                count = _count, states = _stateB, matrices = _matrices,
                sorted = _sortedMatrices, stateStart = _stateStart, stateCount = _stateCount,
            }.Schedule(integrateH).Complete();

            SwapBuffers();
        }

        // Swap the ping-pong pair: the processed cell-sorted B buffers become canonical for next frame.
        void SwapBuffers()
        {
            (_positions, _posB) = (_posB, _positions);
            (_velocities, _velB) = (_velB, _velocities);
            (_states, _stateB) = (_stateB, _states);
            (_stateTimer, _timerB) = (_timerB, _stateTimer);
        }

        void RunManaged(SteeringParams p, float dt)
        {
            int cellCount = math.max(1, p.gridDim.x * p.gridDim.y * p.gridDim.z);
            EnsureGridArrays(cellCount);

            _steerTimer.Restart();

            for (int i = 0; i < _count; i++)
                FishSteering.BuildCellIndex(i, _positions, p, _cellIndex);

            FishSteering.CountingSort(_count, cellCount, _cellIndex, _cellCounts, _cellStart,
                _positions, _velocities, _states, _stateTimer, _posB, _velB, _stateB, _timerB);

            for (int i = 0; i < _count; i++)
            {
                FishSteering.Step(i, _posB, _velB, _stateB, _predatorPositions, _predatorCount,
                    _nodePositions, _nodeCount, _cellStart, useGrid, p, _accelerations, _nearestNeighborSqr, _predatorInRange);
            }
            _steerTimer.Stop();
            _lastSteerMs = _steerTimer.Elapsed.TotalMilliseconds;

            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            int frameBucket = Time.frameCount % buckets;
            float tickDt = dt * buckets;
            uint seed = (uint)(Time.frameCount * 9781 + 1);
            for (int i = frameBucket; i < _count; i += buckets)
            {
                FishSteering.TickBehavior(i, _posB, _nearestNeighborSqr, _predatorInRange,
                    _nodePositions, _nodeCount, _stateB, _timerB, p, tickDt, seed);
            }

            for (int i = 0; i < _count; i++)
            {
                FishSteering.Integrate(i, _posB, _velB, _accelerations, _matrices, dt, p, fishScale);
            }

            FishSteering.BucketByState(_count, _stateB, _matrices, _sortedMatrices, _stateStart, _stateCount);

            SwapBuffers();
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

        void OnDestroy()
        {
            DisposeArrays();
            if (_stateMaterials != null)
                foreach (var m in _stateMaterials)
                    if (m != null) Destroy(m);
            if (_ownsConfig && config != null) Destroy(config);
        }
    }
}
