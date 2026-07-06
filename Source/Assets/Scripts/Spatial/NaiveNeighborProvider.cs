using System.Collections.Generic;
using UnityEngine;

namespace FishSwarm.Spatial
{
    /// <summary>
    /// Brute-force all-pairs neighbour lookup — every query scans every fish, so the whole
    /// frame is O(n²). This is the baseline the spatial grid is measured against; the whole
    /// point of the benchmark is to make its cost visibly explode as the population grows.
    /// </summary>
    public class NaiveNeighborProvider : INeighborProvider
    {
        public string Name => "Naive O(n^2)";

        Vector3[] _positions;
        int _count;
        float _radiusSqr;
        int _maxNeighbors;

        public void Build(Vector3[] positions, int count, float queryRadius, int maxNeighbors)
        {
            _positions = positions;
            _count = count;
            _radiusSqr = queryRadius * queryRadius;
            _maxNeighbors = maxNeighbors;
        }

        public void Query(int index, List<int> results)
        {
            results.Clear();
            Vector3 p = _positions[index];
            // Deliberately scans ALL fish even once the neighbour cap is reached — this is the
            // O(n²) baseline, so its cost must stay proportional to the population, not early-exit.
            for (int j = 0; j < _count; j++)
            {
                if (j == index) continue;
                if ((_positions[j] - p).sqrMagnitude < _radiusSqr && results.Count < _maxNeighbors)
                    results.Add(j);
            }
        }
    }
}
