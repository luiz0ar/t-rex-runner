import { describe, it, expect } from 'vitest';
import { NeuralNetwork, GeneticEvolution } from '../application/genetic-policy.js';

describe('NeuralNetwork', () => {
    it('should initialize weights and biases with random values between -1 and 1', () => {
        const net = new NeuralNetwork(3, 4, 2);
        expect(net.weights1.length).toBe(4);
        expect(net.weights1[0].length).toBe(3);
        expect(net.weights2.length).toBe(2);
        expect(net.weights2[0].length).toBe(4);
        expect(net.bias1.length).toBe(4);
        expect(net.bias2.length).toBe(2);

        // Check range
        net.weights1.forEach(row => {
            row.forEach(w => {
                expect(w).toBeGreaterThanOrEqual(-1);
                expect(w).toBeLessThanOrEqual(1);
            });
        });
    });

    it('should calculate sigmoid feedforward prediction correctly', () => {
        const net = new NeuralNetwork(2, 2, 2);
        const inputs = [0.5, 0.8];
        const outputs = net.predict(inputs);
        expect(outputs.length).toBe(2);
        outputs.forEach(o => {
            expect(o).toBeGreaterThan(0);
            expect(o).toBeLessThan(1);
        });
    });

    it('should clone and copy weights and biases correctly', () => {
        const net = new NeuralNetwork(2, 2, 2);
        const clone = net.clone();
        expect(clone.weights1).toEqual(net.weights1);
        expect(clone.weights2).toEqual(net.weights2);
        expect(clone.bias1).toEqual(net.bias1);
        expect(clone.bias2).toEqual(net.bias2);
    });

    it('should mutate values with a mutation rate', () => {
        const net = new NeuralNetwork(2, 2, 2);
        const oldWeights = JSON.parse(JSON.stringify(net.weights1));
        
        // Mutate with 100% mutation rate
        net.mutate(1.0);
        
        expect(net.weights1).not.toEqual(oldWeights);
    });
});

describe('GeneticEvolution', () => {
    it('should initialize population of correct size', () => {
        const popSize = 5;
        const evolution = new GeneticEvolution(popSize);
        expect(evolution.population.length).toBe(popSize);
        expect(evolution.generation).toBe(1);
        expect(evolution.currentAgentIndex).toBe(0);
    });

    it('should advance indexes and breed next generation on completing population scores', () => {
        const evolution = new GeneticEvolution(4);
        
        // First dino
        expect(evolution.currentAgentIndex).toBe(0);
        let genDone = evolution.registerScore(10);
        expect(genDone).toBe(false);
        expect(evolution.currentAgentIndex).toBe(1);

        // Second, third, fourth dinos
        evolution.registerScore(20);
        evolution.registerScore(50);
        genDone = evolution.registerScore(30);

        // Fourth dino should trigger generation complete and breed
        expect(genDone).toBe(true);
        expect(evolution.generation).toBe(2);
        expect(evolution.currentAgentIndex).toBe(0);
    });
});
