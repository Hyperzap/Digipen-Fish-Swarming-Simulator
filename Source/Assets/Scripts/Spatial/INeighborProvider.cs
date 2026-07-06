using System.Collections.Generic;
using UnityEngine;

namespace FishSwarm.Spatial
{
    /// <summary>
    /// Abstraction over neighbour lookups so the flocking loop is byte-for-byte identical
    /// whether neighbours come from a brute-force O(n²) scan or a spatial grid. This is the seam
    /// the benchmark flips at runtime to show the scaling win on the same live population — both
    /// implementations must return the same neighbour set, so only performance changes.
    /// </summary>
    public interface INeighborProvider
    {
        /// <summary>Human-readable name shown in the benchmark overlay.</summary>
        string Name { get; }

        /// <summary>
        /// Prepare for this frame's queries. <paramref name="positions"/> holds the current
        /// positions of the first <paramref name="count"/> fish; <paramref name="queryRadius"/>
        /// is the perception radius every query will use this frame; <paramref name="maxNeighbors"/>
        /// caps how many neighbours a query returns (the grid returns the nearest ones).
        /// </summary>
        void Build(Vector3[] positions, int count, float queryRadius, int maxNeighbors);

        /// <summary>
        /// Clear <paramref name="results"/> and fill it with up to <c>maxNeighbors</c> indices of
        /// fish within the query radius of fish <paramref name="index"/> (excluding itself).
        /// </summary>
        void Query(int index, List<int> results);
    }
}
