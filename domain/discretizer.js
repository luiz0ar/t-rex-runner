/**
 * Domain service to discretize the continuous game state into a finite number of state buckets.
 * This simplifies the state space for the Q-Learning policy.
 */

/**
 * Discretizes a raw state object into a unique string key representing a state.
 * Refines buckets and explicitly incorporates speed categories to account for jump physics.
 * 
 * @param {Object} rawState - Raw state returned by GameAdapter.readRawState()
 * @returns {string} The discretized state key (e.g. "d3_medium_ground_low_on_ground")
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
    const normalizedDistance = distance / gameSpeed;

    // Discretize distance with higher precision (smaller intervals)
    let distBucket;
    if (normalizedDistance < 0) {
        distBucket = 'passed';
    } else if (normalizedDistance < 3) {
        distBucket = 'd0';
    } else if (normalizedDistance < 6) {
        distBucket = 'd1';
    } else if (normalizedDistance < 9) {
        distBucket = 'd2';
    } else if (normalizedDistance < 12) {
        distBucket = 'd3';
    } else if (normalizedDistance < 16) {
        distBucket = 'd4';
    } else if (normalizedDistance < 20) {
        distBucket = 'd5';
    } else if (normalizedDistance < 25) {
        distBucket = 'd6';
    } else {
        distBucket = 'far';
    }

    // Discretize speed because T-Rex jump duration (air time) is constant.
    // At higher speeds, the T-Rex travels much further horizontally during a jump,
    // requiring the jump to start at a different normalized distance.
    let speedBucket;
    if (gameSpeed < 8) {
        speedBucket = 'slow';
    } else if (gameSpeed < 10) {
        speedBucket = 'medium';
    } else if (gameSpeed < 12) {
        speedBucket = 'fast';
    } else {
        speedBucket = 'hyper';
    }

    // Discretize obstacle type / vertical position
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

    return `${distBucket}_${speedBucket}_${obsBucket}_${airBucket}`;
}
