import { describe, it, expect } from 'vitest';
import { discretizeState } from '../domain/discretizer.js';

describe('discretizeState', () => {
    it('should return NO_STATE when rawState is null or undefined', () => {
        expect(discretizeState(null)).toBe('NO_STATE');
        expect(discretizeState(undefined)).toBe('NO_STATE');
    });

    it('should return NO_OBSTACLE when there is no nextObstacle', () => {
        const rawState = {
            gameSpeed: 10,
            tRex: { x: 50, y: 0, jumping: false },
            nextObstacle: null
        };
        expect(discretizeState(rawState)).toBe('NO_OBSTACLE');
    });

    it('should discretize ground cactus obstacles correctly based on normalized distance and speed', () => {
        const createRawState = (obsX, speed) => ({
            gameSpeed: speed,
            tRex: { x: 50, y: 0, jumping: false },
            nextObstacle: {
                x: obsX,
                y: 105,
                width: 20,
                type: 'CACTUS_SMALL'
            }
        });

        // T-Rex front = 50 + 44 = 94.
        
        // d1 bucket (3 <= normalizedDistance < 6) and medium speed (8 <= speed < 10)
        // distance = 130 - 94 = 36. normalizedDistance = 36 / 9 = 4.0. speed = 9
        expect(discretizeState(createRawState(130, 9))).toBe('d1_medium_ground_low_on_ground');

        // d2 bucket (6 <= normalizedDistance < 9) and fast speed (10 <= speed < 12)
        // distance = 180 - 94 = 86. normalizedDistance = 86 / 10 = 8.6. speed = 10
        expect(discretizeState(createRawState(180, 10))).toBe('d2_fast_ground_low_on_ground');

        // d4 bucket (12 <= normalizedDistance < 16)
        // distance = 250 - 94 = 156. normalizedDistance = 156 / 10 = 15.6. speed = 10
        expect(discretizeState(createRawState(250, 10))).toBe('d4_fast_ground_low_on_ground');

        // far bucket (normalizedDistance >= 25)
        // distance = 350 - 94 = 256. normalizedDistance = 25.6. speed = 10
        expect(discretizeState(createRawState(350, 10))).toBe('far_fast_ground_low_on_ground');

        // passed bucket (normalizedDistance < 0)
        // distance = 80 - 94 = -14. normalizedDistance = -1.4. speed = 10
        expect(discretizeState(createRawState(80, 10))).toBe('passed_fast_ground_low_on_ground');
    });

    it('should discretize pterodactyls (birds) heights correctly', () => {
        const createBirdState = (yPos) => ({
            gameSpeed: 10,
            tRex: { x: 50, y: 0, jumping: false },
            nextObstacle: {
                x: 180, // d2 distance (8.6 normalized)
                y: yPos,
                width: 46,
                type: 'PTERODACTYL'
            }
        });

        // High bird (y <= 50)
        expect(discretizeState(createBirdState(50))).toBe('d2_fast_bird_high_on_ground');
        
        // Mid bird (50 < y <= 75)
        expect(discretizeState(createBirdState(75))).toBe('d2_fast_bird_mid_on_ground');

        // Low bird (y > 75)
        expect(discretizeState(createBirdState(100))).toBe('d2_fast_bird_low_on_ground');
    });

    it('should reflect T-Rex air status in the air bucket', () => {
        const rawState = {
            gameSpeed: 10,
            tRex: { x: 50, y: 25, jumping: true },
            nextObstacle: {
                x: 180,
                y: 105,
                width: 20,
                type: 'CACTUS_SMALL'
            }
        };

        expect(discretizeState(rawState)).toBe('d2_fast_ground_low_in_air');
    });
});
