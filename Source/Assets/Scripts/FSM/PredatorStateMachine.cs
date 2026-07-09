using FishSwarm.Predators;

namespace FishSwarm.FSM
{
    public struct PredatorStateInputs
    {
        public bool HasPreyTarget;
        public float DistanceToPrey;
    }

    public struct PredatorStateSettings
    {
        public float DetectionRadius;
        public float LungeRange;
        public float LungeDuration;
        public float RecoverDuration;
    }

    /// <summary>
    /// Finite state machine for predator intent. Movement code decides how each state moves.
    /// </summary>
    public static class PredatorStateMachine
    {
        public static PredatorState Tick(
            PredatorState current,
            PredatorStateInputs input,
            PredatorStateSettings settings,
            float dt,
            ref float stateTimer)
        {
            stateTimer += dt;

            switch (current)
            {
                case PredatorState.Lunge:
                    if (stateTimer >= settings.LungeDuration)
                        return Enter(current, PredatorState.Recover, ref stateTimer);
                    return PredatorState.Lunge;

                case PredatorState.Recover:
                    if (stateTimer < settings.RecoverDuration)
                        return PredatorState.Recover;
                    return CanSeePrey(input, settings)
                        ? Enter(current, PredatorState.Chase, ref stateTimer)
                        : Enter(current, PredatorState.Patrol, ref stateTimer);

                case PredatorState.Chase:
                    if (!CanSeePrey(input, settings))
                        return Enter(current, PredatorState.Patrol, ref stateTimer);
                    if (input.DistanceToPrey <= settings.LungeRange)
                        return Enter(current, PredatorState.Lunge, ref stateTimer);
                    return PredatorState.Chase;

                case PredatorState.Patrol:
                default:
                    return CanSeePrey(input, settings)
                        ? Enter(current, PredatorState.Chase, ref stateTimer)
                        : PredatorState.Patrol;
            }
        }

        static bool CanSeePrey(PredatorStateInputs input, PredatorStateSettings settings)
        {
            return input.HasPreyTarget && input.DistanceToPrey <= settings.DetectionRadius;
        }

        static PredatorState Enter(PredatorState current, PredatorState next, ref float stateTimer)
        {
            if (current != next)
                stateTimer = 0f;
            return next;
        }
    }
}
