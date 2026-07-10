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
        [Tooltip("Custom shader for the GPU-indirect path (FishSwarm/FishIndirect). Reads per-instance " +
                 "data from a StructuredBuffer and builds the transform on the GPU.")]
        public Shader indirectShader;
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
        [Tooltip("GPU-indirect rendering (transform built on the GPU) vs CPU-matrix instancing.")]
        public bool useIndirectRender = false;
        public Key indirectToggleKey = Key.G;
        [Tooltip("Simulation LOD: distant fish re-run the neighbour scan every Nth frame (sim-only).")]
        public bool useLod = false;
        public Key lodToggleKey = Key.L;
        public bool showStats = true;

        // Simulation state (NativeArrays, Persistent).
        NativeArray<float3> _positions, _velocities, _accelerations;
        NativeArray<FishState> _states;
        NativeArray<float> _stateTimer, _nearestNeighborSqr;
        NativeArray<byte> _predatorInRange;
        NativeArray<byte> _steered; // per-slot this frame: 1 = full scan ran, 0 = LOD-skipped
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
        // Accelerations ping-pong too: LOD-skipped fish reuse their own carried value, and slots
        // reshuffle every frame, so acceleration must travel with the fish through the sort.
        NativeArray<float3> _accelB;
        Camera _camera;

        // GPU-indirect render path: per-instance data uploaded to a GraphicsBuffer, drawn with
        // RenderMeshIndirect + the custom shader. FishInstance is 2×float4 = 32 bytes.
        const int InstanceStride = 32;
        NativeArray<FishInstance> _instances;
        GraphicsBuffer _instanceBuffer, _argsBuffer;
        GraphicsBuffer.IndirectDrawIndexedArgs[] _argsData = new GraphicsBuffer.IndirectDrawIndexedArgs[1];
        Material _indirectMaterial;
        RenderParams _indirectRenderParams;

        int _count, _predatorCount, _nodeCount;
        bool _allocated;

        // Indirect only when explicitly enabled AND the material compiled.
        bool RenderIndirect => useIndirectRender && _indirectMaterial != null;

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
        public bool UseLod => useLod;
        public string ActiveModeName => useGrid ? "Uniform Grid" : "Naive O(n^2)";
        public string RenderModeName => RenderIndirect ? "GPU Indirect" : "Instanced";
        public double LastSimMs => _lastSimMs;
        public double LastNeighborQueryMs => _lastSteerMs;
        public FishConfig Config => config;

        /// <summary>Request a new population; applied at the top of the next Update (race-free).</summary>
        public void RequestPopulation(int n) => _pendingCount = Mathf.Max(0, n);

        /// <summary>Toggle helpers so UI buttons can mirror the B / Tab keys.</summary>
        public void SetBurst(bool on) => useBurst = on;
        public void SetGrid(bool on) => useGrid = on;
        public void SetLod(bool on) => useLod = on;

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
            _steered = new NativeArray<byte>(cap, Allocator.Persistent);
            _matrices = new NativeArray<float4x4>(cap, Allocator.Persistent);
            _sortedMatrices = new NativeArray<float4x4>(cap, Allocator.Persistent);
            _stateStart = new NativeArray<int>(StateCount, Allocator.Persistent);
            _stateCount = new NativeArray<int>(StateCount, Allocator.Persistent);
            _predatorPositions = new NativeArray<float3>(8, Allocator.Persistent);
            _nodePositions = new NativeArray<float3>(8, Allocator.Persistent);

            _posB = new NativeArray<float3>(cap, Allocator.Persistent);
            _velB = new NativeArray<float3>(cap, Allocator.Persistent);
            _accelB = new NativeArray<float3>(cap, Allocator.Persistent);
            _stateB = new NativeArray<FishState>(cap, Allocator.Persistent);
            _timerB = new NativeArray<float>(cap, Allocator.Persistent);
            _cellIndex = new NativeArray<int>(cap, Allocator.Persistent);

            // Only gridDim is needed here (for cell-array sizing), so LOD/camera args are irrelevant.
            var p0 = SteeringParams.From(config, bounds.Bounds.center, bounds.Bounds.size,
                bounds.Bounds.center, false, 0);
            int cellCount = math.max(1, p0.gridDim.x * p0.gridDim.y * p0.gridDim.z);
            _cellCounts = new NativeArray<int>(cellCount, Allocator.Persistent);
            _cellStart = new NativeArray<int>(cellCount + 1, Allocator.Persistent);

            _instances = new NativeArray<FishInstance>(cap, Allocator.Persistent);
            _instanceBuffer?.Release();
            _instanceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, cap, InstanceStride);
            if (_indirectMaterial != null) _indirectMaterial.SetBuffer("_Instances", _instanceBuffer);

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
            _predatorInRange.Dispose(); _steered.Dispose(); _matrices.Dispose(); _sortedMatrices.Dispose();
            _stateStart.Dispose(); _stateCount.Dispose();
            _predatorPositions.Dispose(); _nodePositions.Dispose();
            _posB.Dispose(); _velB.Dispose(); _stateB.Dispose(); _timerB.Dispose();
            _accelB.Dispose();
            _cellIndex.Dispose();
            if (_cellCounts.IsCreated) _cellCounts.Dispose();
            if (_cellStart.IsCreated) _cellStart.Dispose();
            _instances.Dispose();
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

            // GPU-indirect material (custom shader). Optional — the render path falls back to
            // instanced if the shader is missing, so this can never break the scene.
            if (indirectShader != null)
            {
                _indirectMaterial = new Material(indirectShader);
                _indirectMaterial.SetVector("_FishScale", new Vector4(fishScale.x, fishScale.y, fishScale.z, 0f));
                var cols = new Vector4[StateCount];
                for (int s = 0; s < StateCount; s++)
                {
                    Color c = (stateColors != null && s < stateColors.Length) ? stateColors[s] : Color.white;
                    // SetVectorArray does no colour-space conversion, so convert sRGB -> linear
                    // ourselves. Without this the colours read far too bright (washed out) in the
                    // project's Linear colour space, unlike URP/Lit which converts automatically.
                    cols[s] = c.linear;
                }
                _indirectMaterial.SetVectorArray("_StateColors", cols);
                _indirectRenderParams = new RenderParams(_indirectMaterial)
                {
                    worldBounds = worldBounds,
                    shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
                    receiveShadows = false,
                };
            }
            _argsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
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

            if (_camera == null) _camera = Camera.main;
            float3 camPos = _camera != null ? (float3)_camera.transform.position : (float3)bounds.Bounds.center;

            var p = SteeringParams.From(config, bounds.Bounds.center, bounds.Bounds.size,
                camPos, useLod, Time.frameCount);
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
                srcAccel = _accelerations,
                dstPos = _posB, dstVel = _velB, dstState = _stateB, dstTimer = _timerB,
                dstAccel = _accelB,
            }.Schedule(indexH);

            // Steering over the cell-sorted buffers — contiguous neighbour reads. Writes into the
            // sorted accel buffer, which the sort pre-filled with each fish's carried value (LOD).
            var steerH = new SteeringJob
            {
                positions = _posB, velocities = _velB, states = _stateB,
                predators = _predatorPositions, predatorCount = _predatorCount,
                nodes = _nodePositions, nodeCount = _nodeCount,
                cellStart = _cellStart, useGrid = useGrid, p = p,
                accelerations = _accelB, nearestNeighborSqr = _nearestNeighborSqr,
                predatorInRange = _predatorInRange, steered = _steered,
            }.Schedule(_count, 64, sortH);
            steerH.Complete();
            _steerTimer.Stop();
            _lastSteerMs = _steerTimer.Elapsed.TotalMilliseconds;

            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            var behaviorH = new BehaviorJob
            {
                positions = _posB, nearestNeighborSqr = _nearestNeighborSqr,
                predatorInRange = _predatorInRange, nodes = _nodePositions, nodeCount = _nodeCount,
                steered = _steered,
                states = _stateB, stateTimer = _timerB, p = p,
                frameBucket = Time.frameCount % buckets, buckets = buckets,
                tickDt = dt * buckets, frameSeed = (uint)(Time.frameCount * 9781 + 1),
            }.Schedule(_count, 64);

            // Integrate must run after behaviour (behaviour reads posB; integrate writes it).
            var integrateH = new IntegrateJob
            {
                positions = _posB, velocities = _velB, accelerations = _accelB,
                dt = dt, p = p,
            }.Schedule(_count, 64, behaviorH);

            // Render prep: pack instances for the GPU (indirect, no CPU matrices) OR build + bucket
            // matrices for RenderMeshInstanced.
            if (RenderIndirect)
            {
                new PackInstancesJob
                {
                    positions = _posB, velocities = _velB, states = _stateB, instances = _instances,
                }.Schedule(_count, 64, integrateH).Complete();
            }
            else
            {
                var matricesH = new BuildMatricesJob
                {
                    positions = _posB, velocities = _velB, matrices = _matrices, scale = fishScale,
                }.Schedule(_count, 64, integrateH);

                new BucketByStateJob
                {
                    count = _count, states = _stateB, matrices = _matrices,
                    sorted = _sortedMatrices, stateStart = _stateStart, stateCount = _stateCount,
                }.Schedule(matricesH).Complete();
            }

            SwapBuffers();
        }

        // Swap the ping-pong pair: the processed cell-sorted B buffers become canonical for next frame.
        void SwapBuffers()
        {
            (_positions, _posB) = (_posB, _positions);
            (_velocities, _velB) = (_velB, _velocities);
            (_states, _stateB) = (_stateB, _states);
            (_stateTimer, _timerB) = (_timerB, _stateTimer);
            (_accelerations, _accelB) = (_accelB, _accelerations);
        }

        void RunManaged(SteeringParams p, float dt)
        {
            int cellCount = math.max(1, p.gridDim.x * p.gridDim.y * p.gridDim.z);
            EnsureGridArrays(cellCount);

            _steerTimer.Restart();

            for (int i = 0; i < _count; i++)
                FishSteering.BuildCellIndex(i, _positions, p, _cellIndex);

            FishSteering.CountingSort(_count, cellCount, _cellIndex, _cellCounts, _cellStart,
                _positions, _velocities, _states, _stateTimer, _accelerations,
                _posB, _velB, _stateB, _timerB, _accelB);

            for (int i = 0; i < _count; i++)
            {
                FishSteering.Step(i, _posB, _velB, _stateB, _predatorPositions, _predatorCount,
                    _nodePositions, _nodeCount, _cellStart, useGrid, p, _accelB, _nearestNeighborSqr, _predatorInRange, _steered);
            }
            _steerTimer.Stop();
            _lastSteerMs = _steerTimer.Elapsed.TotalMilliseconds;

            int buckets = Mathf.Max(1, config.behaviorTickBuckets);
            int frameBucket = Time.frameCount % buckets;
            float tickDt = dt * buckets;
            uint seed = (uint)(Time.frameCount * 9781 + 1);
            for (int i = frameBucket; i < _count; i += buckets)
            {
                if (_steered[i] == 0) continue; // LOD-skipped: inputs are stale, hold state
                FishSteering.TickBehavior(i, _posB, _nearestNeighborSqr, _predatorInRange,
                    _nodePositions, _nodeCount, _stateB, _timerB, p, tickDt, seed);
            }

            for (int i = 0; i < _count; i++)
            {
                FishSteering.Integrate(i, _posB, _velB, _accelB, dt, p);
            }

            if (RenderIndirect)
            {
                for (int i = 0; i < _count; i++)
                    FishSteering.PackInstance(i, _posB, _velB, _stateB, _instances);
            }
            else
            {
                for (int i = 0; i < _count; i++)
                    FishSteering.BuildMatrix(i, _posB, _velB, _matrices, fishScale);
                FishSteering.BucketByState(_count, _stateB, _matrices, _sortedMatrices, _stateStart, _stateCount);
            }

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
            if (RenderIndirect) RenderFishIndirect();
            else RenderFishInstanced();
        }

        void RenderFishInstanced()
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

        void RenderFishIndirect()
        {
            _instanceBuffer.SetData(_instances, 0, 0, _count);

            _argsData[0].indexCountPerInstance = _fishMesh.GetIndexCount(0);
            _argsData[0].instanceCount = (uint)_count;
            _argsData[0].startIndex = _fishMesh.GetIndexStart(0);
            _argsData[0].baseVertexIndex = _fishMesh.GetBaseVertex(0);
            _argsData[0].startInstance = 0;
            _argsBuffer.SetData(_argsData);

            Graphics.RenderMeshIndirect(_indirectRenderParams, _fishMesh, _argsBuffer, 1);
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
            if (kb[indirectToggleKey].wasPressedThisFrame && _indirectMaterial != null)
                useIndirectRender = !useIndirectRender;
            if (kb[lodToggleKey].wasPressedThisFrame) useLod = !useLod;
        }

        void OnDestroy()
        {
            DisposeArrays();
            if (_stateMaterials != null)
                foreach (var m in _stateMaterials)
                    if (m != null) Destroy(m);
            if (_ownsConfig && config != null) Destroy(config);
            _instanceBuffer?.Release();
            _argsBuffer?.Release();
            if (_indirectMaterial != null) Destroy(_indirectMaterial);
        }
    }
}
