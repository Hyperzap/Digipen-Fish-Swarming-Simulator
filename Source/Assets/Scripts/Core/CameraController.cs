using UnityEngine;
using UnityEngine.InputSystem;
using FishSwarm.Predators;

public class CameraController : MonoBehaviour
{
    [Header("Free camera")]
    [Tooltip("Movement speed for WASD/Space/Shift controls.")]
    public float moveSpeed = 25f;
    [Tooltip("Speed multiplier while Ctrl is held.")]
    public float fastMultiplier = 2.5f;
    [Tooltip("Mouse-look sensitivity while right mouse is held.")]
    public float lookSensitivity = 0.12f;
    [Tooltip("How far each mouse-wheel scroll step moves the camera along its view direction.")]
    public float scrollZoomSpeed = 20f; // change value if you want the zoom to be faster or slower

    [Header("Predator third-person")]
    [Tooltip("Distance behind the predator while predator view is active.")]
    public float predatorFollowDistance = 18f;
    [Tooltip("Height above the predator while predator view is active.")]
    public float predatorFollowHeight = 7f;
    [Tooltip("How quickly the camera catches up to the predator.")]
    public float predatorFollowSharpness = 8f;
    [Tooltip("How quickly the inferred predator heading turns. Lower values reduce wobble.")]
    public float predatorHeadingSharpness = 4f;
    [Tooltip("Look slightly ahead of the predator instead of directly at its centre.")]
    public float predatorLookAhead = 8f;

    [Header("Input")]
    public Key predatorViewKey = Key.F;

    float _yaw;
    float _pitch;
    bool _predatorView;
    Vector3 _lastPredatorPosition;
    Vector3 _predatorForward = Vector3.forward;
    bool _hasPredatorHistory;
    bool _ignoreFirstMouseDelta;

    void Start()
    {
        Vector3 euler = transform.rotation.eulerAngles;
        _yaw = euler.y;
        _pitch = NormalizePitch(euler.x);
    }

    void Update()
    {
        Keyboard kb = Keyboard.current;
        if (kb != null && kb[predatorViewKey].wasPressedThisFrame)
            TogglePredatorView();
    }

    void LateUpdate()
    {
        if (_predatorView)
            UpdatePredatorView();
        else
            UpdateFreeCamera();
    }

    void UpdateFreeCamera()
    {
        UpdateMouseLook();
        UpdateMovement();
        UpdateScrollZoom();
    }

    void UpdateMouseLook()
    {
        Mouse mouse = Mouse.current;
        if (mouse == null) return;

        if (mouse.rightButton.wasPressedThisFrame)
        {
            SyncAnglesFromTransform();
            _ignoreFirstMouseDelta = true;
        }

        if (!mouse.rightButton.isPressed) return;

        if (_ignoreFirstMouseDelta)
        {
            _ignoreFirstMouseDelta = false;
            return;
        }

        Vector2 delta = mouse.delta.ReadValue();
        _yaw += delta.x * lookSensitivity;
        _pitch = Mathf.Clamp(_pitch - delta.y * lookSensitivity, -89f, 89f);
        transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
    }

    void UpdateMovement()
    {
        Keyboard kb = Keyboard.current;
        if (kb == null) return;

        Vector3 input = Vector3.zero;
        if (kb.wKey.isPressed) input += Vector3.forward;
        if (kb.sKey.isPressed) input += Vector3.back;
        if (kb.dKey.isPressed) input += Vector3.right;
        if (kb.aKey.isPressed) input += Vector3.left;
        if (kb.spaceKey.isPressed) input += Vector3.up;
        if (kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed) input += Vector3.down;

        if (input.sqrMagnitude < 1e-6f) return;

        float speed = moveSpeed;
        if (kb.leftCtrlKey.isPressed || kb.rightCtrlKey.isPressed)
            speed *= fastMultiplier;

        Vector3 move =
            transform.forward * input.z +
            transform.right * input.x +
            Vector3.up * input.y;

        transform.position += move.normalized * speed * Time.deltaTime;
    }

    void UpdateScrollZoom()
    {
        Mouse mouse = Mouse.current;
        if (mouse == null) return;

        float scroll = mouse.scroll.ReadValue().y;
        if (Mathf.Abs(scroll) < 0.01f) return;

        transform.position += transform.forward * (scroll * scrollZoomSpeed * Time.deltaTime);
    }

    void UpdatePredatorView()
    {
        Predator predator = CurrentPredator();
        if (predator == null)
        {
            _predatorView = false;
            return;
        }

        Vector3 predatorPos = predator.transform.position;
        Vector3 forward = PredatorForward(predatorPos);
        Vector3 desiredPos = predatorPos - forward * predatorFollowDistance + Vector3.up * predatorFollowHeight;
        Vector3 lookTarget = predatorPos + forward * predatorLookAhead;
        Quaternion desiredRot = Quaternion.LookRotation(lookTarget - desiredPos, Vector3.up);
        float t = 1f - Mathf.Exp(-predatorFollowSharpness * Time.deltaTime);

        transform.position = Vector3.Lerp(transform.position, desiredPos, t);
        transform.rotation = Quaternion.Slerp(transform.rotation, desiredRot, t);

        Vector3 euler = transform.rotation.eulerAngles;
        _yaw = euler.y;
        _pitch = NormalizePitch(euler.x);
    }

    Predator CurrentPredator()
    {
        return Predator.All.Count > 0 ? Predator.All[0] : null;
    }

    void TogglePredatorView()
    {
        _predatorView = !_predatorView;
        _hasPredatorHistory = false;
        SyncAnglesFromTransform();
    }

    Vector3 PredatorForward(Vector3 currentPosition)
    {
        Vector3 targetForward = _predatorForward;
        if (_hasPredatorHistory)
        {
            Vector3 delta = currentPosition - _lastPredatorPosition;
            if (delta.sqrMagnitude > 1e-6f)
                targetForward = delta.normalized;
        }
        else
        {
            targetForward = transform.forward.sqrMagnitude > 1e-6f ? transform.forward : Vector3.forward;
        }

        _lastPredatorPosition = currentPosition;
        _hasPredatorHistory = true;

        float t = 1f - Mathf.Exp(-predatorHeadingSharpness * Time.deltaTime);
        _predatorForward = Vector3.Slerp(_predatorForward, targetForward, t).normalized;
        return _predatorForward;
    }

    void SyncAnglesFromTransform()
    {
        Vector3 euler = transform.rotation.eulerAngles;
        _yaw = euler.y;
        _pitch = NormalizePitch(euler.x);
    }

    static float NormalizePitch(float pitch)
    {
        return pitch > 180f ? pitch - 360f : pitch;
    }
}
