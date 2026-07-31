import { describe, it, expect } from 'vitest';
import { QLearningPolicy } from '../application/q-learning-policy.js';

describe('QLearningPolicy', () => {
    it('should initialize with default parameters', () => {
        const policy = new QLearningPolicy();
        expect(policy.learningRate).toBe(0.1);
        expect(policy.discountFactor).toBe(0.9);
        expect(policy.epsilon).toBe(0.1);
        expect(policy.actions).toEqual(['NONE', 'JUMP', 'DUCK']);
        expect(policy.qTable).toEqual({});
    });

    it('should support custom parameters', () => {
        const policy = new QLearningPolicy({
            learningRate: 0.2,
            discountFactor: 0.95,
            epsilon: 0.05,
            actions: ['JUMP', 'NONE']
        });
        expect(policy.learningRate).toBe(0.2);
        expect(policy.discountFactor).toBe(0.95);
        expect(policy.epsilon).toBe(0.05);
        expect(policy.actions).toEqual(['JUMP', 'NONE']);
    });

    it('should choose actions greedily when epsilon is 0', () => {
        const policy = new QLearningPolicy({ epsilon: 0 });
        
        // Mock a state in the Q-table
        policy.qTable['state1'] = {
            'NONE': 0.1,
            'JUMP': 0.8,
            'DUCK': 0.2
        };

        expect(policy.selectAction('state1')).toBe('JUMP');
    });

    it('should perform Q-value updates correctly using Bellman Equation', () => {
        // alpha = 0.5, gamma = 0.8
        const policy = new QLearningPolicy({
            learningRate: 0.5,
            discountFactor: 0.8,
            epsilon: 0
        });

        // Initialize state1 and state2
        policy.qTable['state1'] = { 'NONE': 0.0, 'JUMP': 1.0, 'DUCK': 0.0 };
        policy.qTable['state2'] = { 'NONE': 2.0, 'JUMP': 5.0, 'DUCK': 0.0 }; // max Q(s', a') = 5.0

        // Take action JUMP from state1, receive reward 10, land in state2
        // Target = R + gamma * max Q(s', a') = 10 + 0.8 * 5.0 = 14.0
        // New Q = Q(s, a) + alpha * (Target - Q(s, a)) = 1.0 + 0.5 * (14.0 - 1.0) = 1.0 + 6.5 = 7.5
        policy.update('state1', 'JUMP', 10, 'state2');

        expect(policy.qTable['state1']['JUMP']).toBe(7.5);
    });

    it('should import and export Q-table successfully', () => {
        const policy = new QLearningPolicy();
        policy.qTable = {
            'stateA': { 'NONE': 0.5, 'JUMP': 0.1, 'DUCK': 0.0 }
        };

        const exported = policy.exportTable();
        const newPolicy = new QLearningPolicy();
        newPolicy.importTable(exported);

        expect(newPolicy.qTable).toEqual(policy.qTable);
    });
});
