using FishSwarm.FSM;

namespace FishSwarm.Behavior
{
    /// <summary>
    /// Compatibility wrapper for callers that still use the old behaviour controller name.
    /// The finite state machine logic lives in <see cref="FiniteStateMachine"/>.
    /// </summary>
    public static class FishBehaviorController
    {
        public static FishState Tick(
            FishState current,
            bool predatorInRange,
            float nearestNeighborDistance,
            bool nearForagingNode,
            float rng,
            float regroupNeighborDistance,
            float foragingChance,
            float foragingDuration,
            float tickDt,
            ref float stateTimer)
        {
            var input = new FishStateInputs
            {
                PredatorInRange = predatorInRange ? (byte)1 : (byte)0,
                NearestNeighborDistance = nearestNeighborDistance,
                NearForagingNode = nearForagingNode ? (byte)1 : (byte)0,
                RandomValue = rng,
            };

            var settings = new FishStateSettings
            {
                RegroupNeighborDistance = regroupNeighborDistance,
                ForagingChance = foragingChance,
                ForagingDuration = foragingDuration,
            };

            return FiniteStateMachine.Tick(current, input, settings, tickDt, ref stateTimer);
        }
    }
}
