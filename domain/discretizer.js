/**
 * Domain service to discretize the continuous game state into a finite number of state buckets.
 * This simplifies the state space for the Q-Learning policy.
 */

/**
 * Discretizes a raw state object into a unique string key representing a state.
 * Uses obstacle height, jump phase, speed, and time-to-collision distance.
 * 
 * @param {Object} rawState - Raw state returned by GameAdapter.readRawState()
 * @returns {string} The discretized state key (e.g. "d1_medium_cactus_large_ground")
 */
export function discretizeState(rawState) {
    if (!rawState) return 'NO_STATE';
    
    const { tRex, nextObstacle, gameSpeed } = rawState;

    if (!nextObstacle) {
        return 'NO_OBSTACLE';
    }

    // Calculate distance from T-Rex's front edge to the obstacle's front edge.
    const tRexFront = tRex.x + 44;
    const distance = nextObstacle.x - tRexFront;

    // Normalize distance by speed to get a proxy for time-to-collision (in frames).
    const normalizedDistance = distance / gameSpeed;

    // Distance buckets — 5 levels for faster Q-table convergence
    let distBucket;
    if (normalizedDistance < 0) {
        distBucket = 'passed';
    } else if (normalizedDistance < 4) {
        distBucket = 'd0';  // Critical — must already be acting
    } else if (normalizedDistance < 8) {
        distBucket = 'd1';  // Jump now
    } else if (normalizedDistance < 14) {
        distBucket = 'd2';  // Approaching
    } else if (normalizedDistance < 22) {
        distBucket = 'd3';  // Getting close
    } else {
        distBucket = 'far';
    }

    // Speed bucket — affects jump arc horizontal distance
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

    // Obstacle type with height differentiation for cacti
    let obsBucket;
    if (nextObstacle.type === 'PTERODACTYL') {
        const y = nextObstacle.y;
        if (y <= 50) {
            obsBucket = 'bird_high';
        } else if (y <= 75) {
            obsBucket = 'bird_mid';
        } else {
            obsBucket = 'bird_low';
        }
    } else {
        // Cactus — small cacti need different jump timing than large
        if (nextObstacle.height <= 35) {
            obsBucket = 'cactus_small';
        } else {
            obsBucket = 'cactus_large';
        }
    }

    // T-Rex vertical state — 4 phases for precise jump arc awareness
    let airBucket;
    if (tRex.ducking) {
        airBucket = 'ducking';
    } else if (!tRex.jumping) {
        airBucket = 'ground';
    } else if (tRex.y > tRex.groundY - 20) {
        // Near ground level while jumping (just took off or about to land)
        airBucket = 'low_air';
    } else {
        airBucket = 'high_air';
    }

    return `${distBucket}_${speedBucket}_${obsBucket}_${airBucket}`;
}
