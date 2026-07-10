using UnityEngine;
using UnityEngine.InputSystem;
using FishSwarm.Core;

namespace FishSwarm.Benchmark
{
    /// <summary>
    /// Draggable IMGUI benchmark HUD: a stats + rolling-graph window and a config window for live
    /// tuning (fish-count presets, steering weights, Burst/grid toggles). Reads/writes the
    /// <see cref="FishManager"/> and its <see cref="FishConfig"/> asset — the manager re-snapshots
    /// the config into its Burst params every frame, so slider edits apply next frame. Windows use
    /// <see cref="GUI.DragWindow"/> so they can be moved in a standalone build.
    /// </summary>
    public class BenchmarkHUD : MonoBehaviour
    {
        public FishManager fishManager;
        [Tooltip("Key that shows/hides the config window.")]
        public Key configToggleKey = Key.C;

        const int Samples = 120;
        readonly float[] _frameMs = new float[Samples];
        readonly float[] _simMs = new float[Samples];
        int _head;

        Rect _statsWin = new Rect(10, 10, 340, 250);
        Rect _configWin = new Rect(360, 10, 330, 470);
        bool _showConfig;

        static readonly int[] Presets = { 100, 1000, 10000, 100000 };

        // Config snapshot captured at play start, for the Reset button.
        struct Snapshot
        {
            public float sep, ali, coh, bnd, threat, maxSpeed, perception;
            public int maxNeighbors;
            public Vector3 axis;
        }
        Snapshot _original;
        bool _haveSnapshot;

        Texture2D _tex;
        GUIStyle _label;

        void Awake()
        {
            if (fishManager == null) fishManager = GetComponent<FishManager>();
            if (fishManager == null) fishManager = FindFirstObjectByType<FishManager>();
        }

        void Start() => CaptureSnapshot();

        void Update()
        {
            Keyboard kb = Keyboard.current;
            if (kb != null && kb[configToggleKey].wasPressedThisFrame) _showConfig = !_showConfig;

            _frameMs[_head] = Time.unscaledDeltaTime * 1000f;
            _simMs[_head] = fishManager != null ? (float)fishManager.LastSimMs : 0f;
            _head = (_head + 1) % Samples;
        }

        void CaptureSnapshot()
        {
            var c = fishManager != null ? fishManager.Config : null;
            if (c == null) return;
            _original = new Snapshot
            {
                sep = c.separationWeight, ali = c.alignmentWeight, coh = c.cohesionWeight,
                bnd = c.boundaryWeight, threat = c.threatWeight, maxSpeed = c.maxSpeed,
                perception = c.perceptionRadius, maxNeighbors = c.maxNeighbors,
                axis = c.cohesionAxisWeights,
            };
            _haveSnapshot = true;
        }

        void ApplySnapshot()
        {
            var c = fishManager != null ? fishManager.Config : null;
            if (!_haveSnapshot || c == null) return;
            c.separationWeight = _original.sep; c.alignmentWeight = _original.ali;
            c.cohesionWeight = _original.coh; c.boundaryWeight = _original.bnd;
            c.threatWeight = _original.threat; c.maxSpeed = _original.maxSpeed;
            c.perceptionRadius = _original.perception; c.maxNeighbors = _original.maxNeighbors;
            c.cohesionAxisWeights = _original.axis;
        }

        void OnGUI()
        {
            if (fishManager == null) return;
            if (_label == null) _label = new GUIStyle(GUI.skin.label) { richText = true };

            _statsWin = GUILayout.Window(9001, _statsWin, DrawStatsWindow, "Benchmark");
            if (_showConfig)
                _configWin = GUILayout.Window(9002, _configWin, DrawConfigWindow, "Config  [C to hide]");
        }

        void DrawStatsWindow(int id)
        {
            float frameMs = Time.smoothDeltaTime * 1000f;
            float fps = frameMs > 0f ? 1000f / frameMs : 0f;
            double sim = fishManager.LastSimMs;
            double steer = fishManager.LastNeighborQueryMs;
            double rest = frameMs - sim;

            GUILayout.Label($"Fish: <b>{fishManager.Count:n0}</b>", _label);
            GUILayout.Label($"Burst: <b>{(fishManager.UseBurst ? "ON" : "OFF")}</b>   Mode: <b>{fishManager.ActiveModeName}</b>", _label);
            GUILayout.Label($"Neighbour+steer: <b>{steer:F2} ms</b>", _label);
            GUILayout.Label($"Sim CPU: <b>{sim:F2} ms</b>   Rest: <b>{rest:F2} ms</b>", _label);
            GUILayout.Label($"Frame: <b>{frameMs:F1} ms</b>  (<b>{fps:F0} fps</b>)", _label);

            Rect graph = GUILayoutUtility.GetRect(320, 92);
            DrawGraph(graph);
            GUILayout.Label("<size=10>bars = frame time · cyan = sim CPU · line = 60 fps</size>", _label);

            GUI.DragWindow(new Rect(0, 0, 10000, 20));
        }

        void DrawConfigWindow(int id)
        {
            FishConfig c = fishManager.Config;

            GUILayout.Label("<b>Fish count</b>  (respawns the school)", _label);
            GUILayout.BeginHorizontal();
            foreach (int n in Presets)
                if (GUILayout.Button(n.ToString("n0")))
                    fishManager.RequestPopulation(n);
            GUILayout.EndHorizontal();

            GUILayout.Space(6);
            GUILayout.Label("<b>Simulation</b>", _label);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(fishManager.UseBurst ? "Burst: ON" : "Burst: OFF"))
                fishManager.SetBurst(!fishManager.UseBurst);
            if (GUILayout.Button(fishManager.UseGrid ? "Grid" : "Naive O(n^2)"))
                fishManager.SetGrid(!fishManager.UseGrid);
            GUILayout.EndHorizontal();

            if (c != null)
            {
                GUILayout.Space(6);
                GUILayout.Label("<b>Weights</b>", _label);
                c.separationWeight = Slider("Separation", c.separationWeight, 0f, 5f);
                c.alignmentWeight = Slider("Alignment", c.alignmentWeight, 0f, 5f);
                c.cohesionWeight = Slider("Cohesion", c.cohesionWeight, 0f, 5f);
                c.boundaryWeight = Slider("Boundary", c.boundaryWeight, 0f, 5f);
                c.threatWeight = Slider("Threat", c.threatWeight, 0f, 12f);

                GUILayout.Space(6);
                GUILayout.Label("<b>Movement / perception</b>", _label);
                c.maxSpeed = Slider("Max speed", c.maxSpeed, 1f, 20f);
                c.perceptionRadius = Slider("Perception", c.perceptionRadius, 1f, 15f);
                c.maxNeighbors = Mathf.RoundToInt(Slider("Max neighbours", c.maxNeighbors, 4f, 64f));

                GUILayout.Space(6);
                GUILayout.Label("<b>School shape (cohesion axis)</b>", _label);
                Vector3 a = c.cohesionAxisWeights;
                a.x = Slider("Axis X", a.x, 0.2f, 4f);
                a.y = Slider("Axis Y", a.y, 0.2f, 4f);
                a.z = Slider("Axis Z", a.z, 0.2f, 4f);
                c.cohesionAxisWeights = a;

                GUILayout.Space(8);
                if (GUILayout.Button("Reset weights to start-of-play"))
                    ApplySnapshot();
            }

            GUI.DragWindow(new Rect(0, 0, 10000, 20));
        }

        float Slider(string label, float val, float min, float max)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(110));
            val = GUILayout.HorizontalSlider(val, min, max, GUILayout.Width(135));
            GUILayout.Label(val.ToString("0.00"), _label, GUILayout.Width(42));
            GUILayout.EndHorizontal();
            return val;
        }

        void DrawGraph(Rect r)
        {
            if (_tex == null) _tex = Texture2D.whiteTexture;
            DrawRect(r, new Color(0f, 0f, 0f, 0.55f));

            const float maxMs = 33.34f; // graph ceiling = 30 fps
            float dx = r.width / Samples;
            for (int i = 0; i < Samples; i++)
            {
                int idx = (_head + i) % Samples;
                float fMs = _frameMs[idx];
                float sMs = _simMs[idx];
                float x = r.x + i * dx;
                float w = Mathf.Max(1f, dx);

                float fh = Mathf.Clamp01(fMs / maxMs) * r.height;
                Color col = fMs > 33.34f ? new Color(0.90f, 0.25f, 0.20f)
                          : fMs > 16.67f ? new Color(0.95f, 0.80f, 0.20f)
                                         : new Color(0.35f, 0.80f, 0.40f);
                DrawRect(new Rect(x, r.yMax - fh, w, fh), col);

                float sh = Mathf.Clamp01(sMs / maxMs) * r.height;
                DrawRect(new Rect(x, r.yMax - sh, w, 2f), new Color(0.30f, 0.80f, 0.95f, 0.95f));
            }

            float y60 = r.yMax - (16.67f / maxMs) * r.height;
            DrawRect(new Rect(r.x, y60, r.width, 1f), new Color(1f, 1f, 1f, 0.5f));
        }

        void DrawRect(Rect r, Color c)
        {
            Color prev = GUI.color;
            GUI.color = c;
            GUI.DrawTexture(r, _tex);
            GUI.color = prev;
        }
    }
}
