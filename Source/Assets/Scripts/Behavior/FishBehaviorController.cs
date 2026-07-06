namespace FishSwarm.Behavior
{
    /// <summary>
    /// Lightweight per-fish state machine, evaluated at low frequency (staggered across the school
    /// by the FishManager, ~5–10 Hz). It decides only the <see cref="FishState"/>; the steering
    /// weights each state implies live on the FishConfig, and the actual split/reform emerges from
    /// those weights plus the threat force. Kept as a pure static function (no Unity types) so it
    /// stays trivially testable and allocation-free.
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
            // A predator in range always wins: flee immediately, whatever the current state.
            if (predatorInRange)
            {
                if (current != FishState.Fleeing) stateTimer = 0f;
                return FishState.Fleeing;
            }

            switch (current)
            {
                case FishState.Fleeing:
                    // Predator gone — pull back toward the school.
                    stateTimer = 0f;
                    return FishState.Regrouping;

                case FishState.Regrouping:
                    // Considered rejoined once a neighbour is close again.
                    if (nearestNeighborDistance < regroupNeighborDistance)
                    {
                        stateTimer = 0f;
                        return FishState.Schooling;
                    }
                    return FishState.Regrouping;

                case FishState.Foraging:
                    stateTimer += tickDt;
                    if (stateTimer > foragingDuration || !nearForagingNode)
                    {
                        stateTimer = 0f;
                        return FishState.Schooling;
                    }
                    return FishState.Foraging;

                case FishState.Schooling:
                default:
                    // Unthreatened and near a feeding spot — occasionally break off to forage.
                    if (nearForagingNode && rng < foragingChance)
                    {
                        stateTimer = 0f;
                        return FishState.Foraging;
                    }
                    return FishState.Schooling;
            }
        }
    }
}
