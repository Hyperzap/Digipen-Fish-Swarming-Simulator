using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using FishSwarm.Behavior;

namespace FishSwarm.Jobs
{
    /// <summary>
    /// Thin Burst-compiled job wrappers around the shared <see cref="FishSteering"/> statics. The
    /// Burst-OFF path calls the very same statics from a main-thread loop, so these jobs add
    /// parallelism + Burst codegen without duplicating any maths.
    ///
    /// Grid pipeline: <see cref="BuildCellIndexJob"/> (parallel) -> <see cref="CountingSortJob"/>
    /// (reorders the data into cell order) -> <see cref="SteeringJob"/> (contiguous neighbour scan).
    /// </summary>

    [BurstCompile]
    public struct BuildCellIndexJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        public SteeringParams p;
        [WriteOnly] public NativeArray<int> cellIndex;

        public void Execute(int i) => FishSteering.BuildCellIndex(i, positions, p, cellIndex);
    }

    [BurstCompile]
    public struct CountingSortJob : IJob
    {
        public int count;
        public int cellCount;
        [ReadOnly] public NativeArray<int> cellIndex;
        public NativeArray<int> cellCounts; // scratch (reused as write cursor)
        public NativeArray<int> cellStart;  // out (length cellCount + 1)

        [ReadOnly] public NativeArray<float3> srcPos;
        [ReadOnly] public NativeArray<float3> srcVel;
        [ReadOnly] public NativeArray<FishState> srcState;
        [ReadOnly] public NativeArray<float> srcTimer;

        [WriteOnly] public NativeArray<float3> dstPos;
        [WriteOnly] public NativeArray<float3> dstVel;
        [WriteOnly] public NativeArray<FishState> dstState;
        [WriteOnly] public NativeArray<float> dstTimer;

        public void Execute()
        {
            FishSteering.CountingSort(count, cellCount, cellIndex, cellCounts, cellStart,
                srcPos, srcVel, srcState, srcTimer, dstPos, dstVel, dstState, dstTimer);
        }
    }

    [BurstCompile]
    public struct SteeringJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<FishState> states;
        [ReadOnly] public NativeArray<float3> predators;
        public int predatorCount;
        [ReadOnly] public NativeArray<float3> nodes;
        public int nodeCount;
        [ReadOnly] public NativeArray<int> cellStart;
        public bool useGrid;
        public SteeringParams p;

        [WriteOnly] public NativeArray<float3> accelerations;
        [WriteOnly] public NativeArray<float> nearestNeighborSqr;
        [WriteOnly] public NativeArray<byte> predatorInRange;

        public void Execute(int i)
        {
            FishSteering.Step(i, positions, velocities, states, predators, predatorCount,
                nodes, nodeCount, cellStart, useGrid, p, accelerations, nearestNeighborSqr, predatorInRange);
        }
    }

    [BurstCompile]
    public struct BehaviorJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float> nearestNeighborSqr;
        [ReadOnly] public NativeArray<byte> predatorInRange;
        [ReadOnly] public NativeArray<float3> nodes;
        public int nodeCount;
        public NativeArray<FishState> states;
        public NativeArray<float> stateTimer;
        public SteeringParams p;
        public int frameBucket;
        public int buckets;
        public float tickDt;
        public uint frameSeed;

        public void Execute(int i)
        {
            if ((i % buckets) != frameBucket) return; // staggered: only this frame's slice ticks
            FishSteering.TickBehavior(i, positions, nearestNeighborSqr, predatorInRange,
                nodes, nodeCount, states, stateTimer, p, tickDt, frameSeed);
        }
    }

    [BurstCompile]
    public struct IntegrateJob : IJobParallelFor
    {
        public NativeArray<float3> positions;
        public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float3> accelerations;
        [WriteOnly] public NativeArray<float4x4> matrices;
        public float dt;
        public SteeringParams p;
        public float3 scale;

        public void Execute(int i)
        {
            FishSteering.Integrate(i, positions, velocities, accelerations, matrices, dt, p, scale);
        }
    }

    [BurstCompile]
    public struct BucketByStateJob : IJob
    {
        public int count;
        [ReadOnly] public NativeArray<FishState> states;
        [ReadOnly] public NativeArray<float4x4> matrices;
        [WriteOnly] public NativeArray<float4x4> sorted;
        [WriteOnly] public NativeArray<int> stateStart;
        [WriteOnly] public NativeArray<int> stateCount;

        public void Execute()
        {
            FishSteering.BucketByState(count, states, matrices, sorted, stateStart, stateCount);
        }
    }
}
