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
            tRex: { x: 50, y: 0, jumping: false, ducking: false, groundY: 0 },
            nextObstacle: null
        };
        expect(discretizeState(rawState)).toBe('NO_OBSTACLE');
    });

    it('should discretize ground cactus obstacles correctly based on normalized distance and speed', () => {
        const createRawState = (obsX, speed, obsHeight = 50) => ({
            gameSpeed: speed,
            tRex: { x: 50, y: 93, jumping: false, ducking: false, groundY: 93 },
            nextObstacle: {
                x: obsX,
                y: 105,
                width: 20,
                height: obsHeight,
                type: 'CACTUS_SMALL'
            }
        });

        // T-Rex front = 50 + 44 = 94.

        // d1 bucket (4 <= normalizedDistance < 8) and medium speed (8 <= speed < 10)
        // distance = 130 - 94 = 36. normalizedDistance = 36 / 9 = 4.0. speed = 9
        expect(discretizeState(createRawState(130, 9))).toBe('d1_medium_cactus_large_ground');

        // d2 bucket (8 <= normalizedDistance < 14) and fast speed (10 <= speed < 12)
        // distance = 180 - 94 = 86. normalizedDistance = 86 / 10 = 8.6. speed = 10
        expect(discretizeState(createRawState(180, 10))).toBe('d2_fast_cactus_large_ground');

        // d3 bucket (14 <= normalizedDistance < 22)
        // distance = 250 - 94 = 156. normalizedDistance = 156 / 10 = 15.6. speed = 10
        expect(discretizeState(createRawState(250, 10))).toBe('d3_fast_cactus_large_ground');

        // far bucket (normalizedDistance >= 22)
        // distance = 350 - 94 = 256. normalizedDistance = 25.6. speed = 10
        expect(discretizeState(createRawState(350, 10))).toBe('far_fast_cactus_large_ground');

        // passed bucket (normalizedDistance < 0)
        // distance = 80 - 94 = -14. normalizedDistance = -1.4. speed = 10
        expect(discretizeState(createRawState(80, 10))).toBe('passed_fast_cactus_large_ground');

        // Small cactus (height <= 35) should be classified differently
        expect(discretizeState(createRawState(180, 10, 30))).toBe('d2_fast_cactus_small_ground');
    });

    it('should discretize pterodactyls (birds) heights correctly', () => {
        const createBirdState = (yPos) => ({
            gameSpeed: 10,
            tRex: { x: 50, y: 93, jumping: false, ducking: false, groundY: 93 },
            nextObstacle: {
                x: 180, // d2 distance (8.6 normalized)
                y: yPos,
                width: 46,
                height: 40,
                type: 'PTERODACTYL'
            }
        });

        // High bird (y <= 50)
        expect(discretizeState(createBirdState(50))).toBe('d2_fast_bird_high_ground');
        
        // Mid bird (50 < y <= 75)
        expect(discretizeState(createBirdState(75))).toBe('d2_fast_bird_mid_ground');

        // Low bird (y > 75)
        expect(discretizeState(createBirdState(100))).toBe('d2_fast_bird_low_ground');
    });

    it('should reflect T-Rex vertical state with 4 phases', () => {
        const createState = (y, jumping, ducking = false, groundY = 93) => ({
            gameSpeed: 10,
            tRex: { x: 50, y, jumping, ducking, groundY },
            nextObstacle: {
                x: 180,
                y: 105,
                width: 20,
                height: 50,
                type: 'CACTUS_SMALL'
            }
        });

        // On ground
        expect(discretizeState(createState(93, false))).toBe('d2_fast_cactus_large_ground');

        // Ducking
        expect(discretizeState(createState(93, false, true))).toBe('d2_fast_cactus_large_ducking');

        // Low air (y > groundY - 20, i.e. y > 73)
        expect(discretizeState(createState(80, true))).toBe('d2_fast_cactus_large_low_air');

        // High air (y <= groundY - 20, i.e. y <= 73)
        expect(discretizeState(createState(50, true))).toBe('d2_fast_cactus_large_high_air');
    });
});
