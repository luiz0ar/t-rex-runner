import { describe, it, expect } from 'vitest';
import { NeuralNetwork, GeneticEvolution } from '../application/genetic-policy.js';

describe('NeuralNetwork (Neuroevolution)', () => {
    it('should initialize weights and biases with correct dimensions for 6 inputs, 8 hidden, 3 outputs', () => {
        const net = new NeuralNetwork(6, 8, 3);
        expect(net.weights1.length).toBe(8);
        expect(net.weights1[0].length).toBe(6);
        expect(net.weights2.length).toBe(8);
        expect(net.weights2[0].length).toBe(8);
        expect(net.weights3.length).toBe(3);
        expect(net.weights3[0].length).toBe(8);

        expect(net.bias1.length).toBe(8);
        expect(net.bias2.length).toBe(8);
        expect(net.bias3.length).toBe(3);
    });

    it('should calculate feedforward prediction correctly with sigmoid outputs', () => {
        const net = new NeuralNetwork(6, 8, 3);
        const inputs = [0.2, 0.5, 0.1, 0.0, 0.4, 0.0];
        const outputs = net.predict(inputs);

        expect(outputs.length).toBe(3);
        outputs.forEach(val => {
            expect(val).toBeGreaterThanOrEqual(0);
            expect(val).toBeLessThanOrEqual(1);
        });
    });

    it('should clone weights and biases correctly', () => {
        const net = new NeuralNetwork(6, 8, 3);
        const clone = net.clone();

        expect(clone.weights1).toEqual(net.weights1);
        expect(clone.weights2).toEqual(net.weights2);
        expect(clone.weights3).toEqual(net.weights3);
        expect(clone.bias1).toEqual(net.bias1);
        expect(clone.bias2).toEqual(net.bias2);
        expect(clone.bias3).toEqual(net.bias3);
    });

    it('should mutate weights with Gaussian noise', () => {
        const net = new NeuralNetwork(6, 8, 3);
        const originalWeights = JSON.parse(JSON.stringify(net.weights1));

        // Mutate with 100% mutation rate
        net.mutate(1.0, 0.5);

        expect(net.weights1).not.toEqual(originalWeights);
    });

    it('should perform crossover between two parent networks', () => {
        const parentA = new NeuralNetwork(6, 8, 3);
        const parentB = new NeuralNetwork(6, 8, 3);

        const child = parentA.crossover(parentB);
        expect(child.weights1.length).toBe(8);
        expect(child.weights3.length).toBe(3);
    });
});

describe('GeneticEvolution (Population & Tournament Selection)', () => {
    it('should initialize population of specified size', () => {
        const popSize = 10;
        const evolution = new GeneticEvolution(popSize, 0.10, 0.20);
        expect(evolution.population.length).toBe(popSize);
        expect(evolution.generation).toBe(1);
        expect(evolution.currentAgentIndex).toBe(0);
    });

    it('should select fitter candidates via tournament selection', () => {
        const evolution = new GeneticEvolution(5);
        evolution.population[0].fitness = 10;
        evolution.population[1].fitness = 500; // Strongest
        evolution.population[2].fitness = 20;

        // Run tournament selection multiple times
        let selectedBestCount = 0;
        for (let i = 0; i < 20; i++) {
            const selectedBrain = evolution.tournamentSelect(evolution.population, 3);
            if (selectedBrain === evolution.population[1].brain) {
                selectedBestCount++;
            }
        }
        // Fittest individual (fitness 500) should be selected frequently
        expect(selectedBestCount).toBeGreaterThan(5);
    });

    it('should advance dino index and trigger evolution when generation ends', () => {
        const evolution = new GeneticEvolution(4);

        expect(evolution.currentAgentIndex).toBe(0);
        expect(evolution.registerScore(100)).toBe(false);
        expect(evolution.currentAgentIndex).toBe(1);

        evolution.registerScore(250);
        evolution.registerScore(400);

        const newGenStarted = evolution.registerScore(150);
        expect(newGenStarted).toBe(true);
        expect(evolution.generation).toBe(2);
        expect(evolution.currentAgentIndex).toBe(0);
    });
});
