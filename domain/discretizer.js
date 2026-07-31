/**
 * Domain service to discretize the continuous game state into a finite number of state buckets.
 * This simplifies the state space for the Q-Learning policy.
 */

/**
 * Discretizes a raw state object into a unique string key representing a state.
 * Uses speed-normalized distance (time-to-collision) to ensure training generalizes across game speeds.
 * 
 * @param {Object} rawState - Raw state returned by GameAdapter.readRawState()
 * @returns {string} The discretized state key (e.g. "close_low_ground")
 */
export function discretizeState(rawState) {
    if (!rawState) return 'NO_STATE';
    
    const { tRex, nextObstacle, gameSpeed } = rawState;

    if (!nextObstacle) {
        return 'NO_OBSTACLE';
    }

    // T-Rex width is 44, START_X_POS is 50.
    // Calculate distance from T-Rex's front edge to the obstacle's front edge.
    const tRexFront = tRex.x + 44;
    const distance = nextObstacle.x - tRexFront;

    // Normalize distance by speed to get a proxy for time-to-collision (in frames).
    // This allows the agent to handle increasing speed automatically.
    const normalizedDistance = distance / gameSpeed;

    // Discretize distance
    let distBucket;
    if (normalizedDistance < 0) {
        distBucket = 'passed';
    } else if (normalizedDistance < 5) {
        distBucket = 'danger';
    } else if (normalizedDistance < 12) {
        distBucket = 'close';
    } else if (normalizedDistance < 22) {
        distBucket = 'medium';
    } else {
        distBucket = 'far';
    }

    // Discretize obstacle type / vertical position
    // Cactus is always on the ground.
    // Pterodactyls can fly at different heights (yPos):
    // - 100: low (needs jump)
    // - 75: mid (needs duck)
    // - 50: high (can be ignored or ducked under safely)
    let obsBucket = 'ground_low';
    if (nextObstacle.type === 'PTERODACTYL') {
        const y = nextObstacle.y;
        if (y <= 50) {
            obsBucket = 'bird_high';
        } else if (y <= 75) {
            obsBucket = 'bird_mid';
        } else {
            obsBucket = 'bird_low';
        }
    }

    // T-Rex vertical state
    const airBucket = tRex.jumping ? 'in_air' : 'on_ground';

    return `${distBucket}_${obsBucket}_${airBucket}`;
}
