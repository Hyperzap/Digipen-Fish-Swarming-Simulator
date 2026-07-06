using UnityEngine;
using UnityEngine.InputSystem;

namespace FishSwarm.Core
{
    /// <summary>
    /// Slow orbit camera that frames the ocean volume so the school's split/reform reads
    /// clearly on screen. Purely a viewing aid — no simulation logic lives here. Uses the
    /// new Input System (the project runs with Active Input Handling = Input System Package),
    /// so key checks go through <see cref="Keyboard.current"/> rather than legacy Input.
    /// </summary>
    public class CameraOrbit : MonoBehaviour
    {
        [Tooltip("Point the camera orbits around and looks at. Falls back to world origin if unset.")]
        public Transform target;

        [Tooltip("Horizontal distance from the target.")]
        public float radius = 55f;

        [Tooltip("Height above the target.")]
        public float height = 15f;

        [Tooltip("Orbit speed in degrees per second.")]
        public float degreesPerSecond = 8f;

        float _angle;
        bool _paused;

        void Update()
        {
            Keyboard kb = Keyboard.current;
            if (kb != null && kb.oKey.wasPressedThisFrame)
                _paused = !_paused;

            if (!_paused)
                _angle += degreesPerSecond * Time.deltaTime;

            Vector3 center = target != null ? target.position : Vector3.zero;
            float rad = _angle * Mathf.Deg2Rad;
            Vector3 offset = new Vector3(Mathf.Cos(rad) * radius, height, Mathf.Sin(rad) * radius);
            transform.position = center + offset;
            transform.rotation = Quaternion.LookRotation(center - transform.position, Vector3.up);
        }
    }
}
