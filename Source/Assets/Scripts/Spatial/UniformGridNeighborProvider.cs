using System.Collections.Generic;
using UnityEngine;

namespace FishSwarm.Spatial
{
    /// <summary>
    /// Uniform spatial-hash grid built with a counting sort into flattened arrays (no per-cell
    /// List allocations). Each frame it buckets every fish into a cell, prefix-sums the counts
    /// into per-cell start offsets, then scatters fish indices into one contiguous "sorted"
    /// array. A query then only scans the fish's own cell plus its 26 neighbours.
    ///
    /// Cell size equals the query radius, so the 3×3×3 cell neighbourhood is guaranteed to
    /// contain every fish within the radius. The grid origin/extent track the school's actual
    /// bounding box each frame, keeping the cell count tight as the school moves. This is the
    /// classic CPU layout that ports directly to a GPU counting sort in the scale-up pass.
    /// </summary>
    public class UniformGridNeighborProvider : INeighborProvider
    {
        public string Name => "Uniform Grid";

        Vector3[] _positions;
        int _count;
        float _cellSize = 1f;
        float _radiusSqr;
        int _maxNeighbors;

        Vector3 _min;
        int _dimX, _dimY, _dimZ, _cellCount;

        int[] _counts;    // fish per cell            (length >= cellCount)
        int[] _cellStart; // prefix-summed offsets    (length >= cellCount + 1)
        int[] _cursor;    // scatter write cursor      (length >= cellCount)
        int[] _sorted;    // fish indices, cell-sorted (length >= count)
        int[] _fishCell;  // cached cell index per fish(length >= count)

        public void Build(Vector3[] positions, int count, float queryRadius, int maxNeighbors)
        {
            _positions = positions;
            _count = count;
            _radiusSqr = queryRadius * queryRadius;
            _maxNeighbors = maxNeighbors;
            _cellSize = Mathf.Max(queryRadius, 0.0001f);

            if (count == 0)
            {
                _cellCount = 0;
                return;
            }

            // Bounding box of the live population — keeps the grid no bigger than the school.
            Vector3 min = positions[0];
            Vector3 max = positions[0];
            for (int i = 1; i < count; i++)
            {
                Vector3 p = positions[i];
                if (p.x < min.x) min.x = p.x; else if (p.x > max.x) max.x = p.x;
                if (p.y < min.y) min.y = p.y; else if (p.y > max.y) max.y = p.y;
                if (p.z < min.z) min.z = p.z; else if (p.z > max.z) max.z = p.z;
            }
            _min = min;

            float inv = 1f / _cellSize;
            _dimX = Mathf.Max(1, Mathf.FloorToInt((max.x - min.x) * inv) + 1);
            _dimY = Mathf.Max(1, Mathf.FloorToInt((max.y - min.y) * inv) + 1);
            _dimZ = Mathf.Max(1, Mathf.FloorToInt((max.z - min.z) * inv) + 1);
            _cellCount = _dimX * _dimY * _dimZ;

            EnsureCapacity(count, _cellCount);

            // 1) Count fish per cell (and cache each fish's cell for the scatter pass).
            System.Array.Clear(_counts, 0, _cellCount);
            for (int i = 0; i < count; i++)
            {
                int c = CellIndex(positions[i]);
                _fishCell[i] = c;
                _counts[c]++;
            }

            // 2) Prefix sum -> start offset per cell; seed the scatter cursors.
            int acc = 0;
            for (int c = 0; c < _cellCount; c++)
            {
                _cellStart[c] = acc;
                _cursor[c] = acc;
                acc += _counts[c];
            }
            _cellStart[_cellCount] = acc;

            // 3) Scatter fish indices into contiguous per-cell runs.
            for (int i = 0; i < count; i++)
            {
                int c = _fishCell[i];
                _sorted[_cursor[c]++] = i;
            }
        }

        public void Query(int index, List<int> results)
        {
            results.Clear();
            if (_count == 0) return;

            Vector3 p = _positions[index];
            int cx = CellCoord(p.x, _min.x, _dimX);
            int cy = CellCoord(p.y, _min.y, _dimY);
            int cz = CellCoord(p.z, _min.z, _dimZ);

            // Scan the fish's own cell first: it holds the closest, roughly symmetric neighbours,
            // so when the cap trims the set we keep a nearest-neighbour sample rather than a
            // one-sided one. In a dense school this cell alone usually fills the cap and we return.
            int centre = cx + _dimX * (cy + _dimY * cz);
            if (ScanCell(centre, index, p, results)) return;

            int xLo = Mathf.Max(cx - 1, 0), xHi = Mathf.Min(cx + 1, _dimX - 1);
            int yLo = Mathf.Max(cy - 1, 0), yHi = Mathf.Min(cy + 1, _dimY - 1);
            int zLo = Mathf.Max(cz - 1, 0), zHi = Mathf.Min(cz + 1, _dimZ - 1);

            for (int z = zLo; z <= zHi; z++)
            {
                for (int y = yLo; y <= yHi; y++)
                {
                    int rowBase = _dimX * (y + _dimY * z);
                    for (int x = xLo; x <= xHi; x++)
                    {
                        int c = x + rowBase;
                        if (c == centre) continue; // already scanned
                        if (ScanCell(c, index, p, results)) return;
                    }
                }
            }
        }

        /// <summary>
        /// Add the in-radius fish of cell <paramref name="c"/> to <paramref name="results"/>.
        /// Returns true once the neighbour cap is reached, signalling the query to stop.
        /// </summary>
        bool ScanCell(int c, int index, Vector3 p, List<int> results)
        {
            int start = _cellStart[c];
            int end = _cellStart[c + 1];
            for (int k = start; k < end; k++)
            {
                int j = _sorted[k];
                if (j == index) continue;
                if ((_positions[j] - p).sqrMagnitude < _radiusSqr)
                {
                    results.Add(j);
                    if (results.Count >= _maxNeighbors) return true;
                }
            }
            return false;
        }

        int CellIndex(Vector3 p)
        {
            int x = CellCoord(p.x, _min.x, _dimX);
            int y = CellCoord(p.y, _min.y, _dimY);
            int z = CellCoord(p.z, _min.z, _dimZ);
            return x + _dimX * (y + _dimY * z);
        }

        int CellCoord(float value, float origin, int dim)
        {
            int c = (int)((value - origin) / _cellSize);
            return Mathf.Clamp(c, 0, dim - 1);
        }

        void EnsureCapacity(int count, int cellCount)
        {
            if (_sorted == null || _sorted.Length < count)
            {
                _sorted = new int[count];
                _fishCell = new int[count];
            }
            if (_counts == null || _counts.Length < cellCount)
            {
                _counts = new int[cellCount];
                _cursor = new int[cellCount];
                _cellStart = new int[cellCount + 1];
            }
        }
    }
}
