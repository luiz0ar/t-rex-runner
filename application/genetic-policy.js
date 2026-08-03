/**
 * Neural Network and Genetic Algorithm Policy for Neuroevolution in T-Rex Runner.
 * Features:
 * - Multi-layer Feedforward Neural Network (6 inputs -> 8 hidden -> 8 hidden -> 3 outputs)
 * - Tanh hidden activations and Sigmoid output activations
 * - Elitism, Uniform Crossover, and Gaussian Mutation
 * - Tournament Selection to prevent inbreeding and preserve population diversity
 */

function gaussianRandom(mean = 0, stdev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return num * stdev + mean;
}

export class NeuralNetwork {
    /**
     * @param {number} inputNodes 
     * @param {number} hiddenNodes 
     * @param {number} outputNodes 
     * @param {Object} [weights] 
     * @param {Object} [biases] 
     */
    constructor(inputNodes = 6, hiddenNodes = 8, outputNodes = 3, weights = null, biases = null) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        if (weights && biases) {
            this.weights1 = JSON.parse(JSON.stringify(weights.w1));
            this.weights2 = JSON.parse(JSON.stringify(weights.w2));
            this.weights3 = JSON.parse(JSON.stringify(weights.w3));
            this.bias1 = JSON.parse(JSON.stringify(biases.b1));
            this.bias2 = JSON.parse(JSON.stringify(biases.b2));
            this.bias3 = JSON.parse(JSON.stringify(biases.b3));
        } else {
            // Xavier / Glorot-like random initialization
            const initScale1 = Math.sqrt(2 / (this.inputNodes + this.hiddenNodes));
            const initScale2 = Math.sqrt(2 / (this.hiddenNodes + this.hiddenNodes));
            const initScale3 = Math.sqrt(2 / (this.hiddenNodes + this.outputNodes));

            this.weights1 = Array.from({ length: this.hiddenNodes }, () =>
                Array.from({ length: this.inputNodes }, () => gaussianRandom(0, initScale1))
            );
            this.weights2 = Array.from({ length: this.hiddenNodes }, () =>
                Array.from({ length: this.hiddenNodes }, () => gaussianRandom(0, initScale2))
            );
            this.weights3 = Array.from({ length: this.outputNodes }, () =>
                Array.from({ length: this.hiddenNodes }, () => gaussianRandom(0, initScale3))
            );

            this.bias1 = Array.from({ length: this.hiddenNodes }, () => 0);
            this.bias2 = Array.from({ length: this.hiddenNodes }, () => 0);
            this.bias3 = Array.from({ length: this.outputNodes }, () => 0);
        }
    }

    _tanh(x) {
        return Math.tanh(x);
    }

    _sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Feeds inputs forward through the 2-hidden layer network.
     * @param {Array<number>} inputs 
     * @returns {Array<number>} Outputs [NONE, JUMP, DUCK]
     */
    predict(inputs) {
        // Hidden Layer 1 (Tanh)
        const h1 = [];
        for (let i = 0; i < this.hiddenNodes; i++) {
            let sum = this.bias1[i];
            for (let j = 0; j < this.inputNodes; j++) {
                sum += inputs[j] * this.weights1[i][j];
            }
            h1.push(this._tanh(sum));
        }

        // Hidden Layer 2 (Tanh)
        const h2 = [];
        for (let i = 0; i < this.hiddenNodes; i++) {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenNodes; j++) {
                sum += h1[j] * this.weights2[i][j];
            }
            h2.push(this._tanh(sum));
        }

        // Output Layer (Sigmoid)
        const outputs = [];
        for (let i = 0; i < this.outputNodes; i++) {
            let sum = this.bias3[i];
            for (let j = 0; j < this.hiddenNodes; j++) {
                sum += h2[j] * this.weights3[i][j];
            }
            outputs.push(this._sigmoid(sum));
        }

        return outputs;
    }

    /**
     * Mutates weights and biases using Gaussian noise.
     * @param {number} rate - Probability of mutating a parameter
     * @param {number} strength - Standard deviation of Gaussian noise
     */
    mutate(rate = 0.10, strength = 0.20) {
        const mutateVal = (val) => {
            if (Math.random() < rate) {
                return val + gaussianRandom(0, strength);
            }
            return val;
        };

        this.weights1 = this.weights1.map(row => row.map(mutateVal));
        this.weights2 = this.weights2.map(row => row.map(mutateVal));
        this.weights3 = this.weights3.map(row => row.map(mutateVal));
        this.bias1 = this.bias1.map(mutateVal);
        this.bias2 = this.bias2.map(mutateVal);
        this.bias3 = this.bias3.map(mutateVal);
    }

    /**
     * Uniform crossover with a partner network.
     * @param {NeuralNetwork} partner 
     * @returns {NeuralNetwork}
     */
    crossover(partner) {
        const child = new NeuralNetwork(this.inputNodes, this.hiddenNodes, this.outputNodes);

        child.weights1 = this.weights1.map((row, i) =>
            row.map((val, j) => Math.random() < 0.5 ? val : partner.weights1[i][j])
        );
        child.weights2 = this.weights2.map((row, i) =>
            row.map((val, j) => Math.random() < 0.5 ? val : partner.weights2[i][j])
        );
        child.weights3 = this.weights3.map((row, i) =>
            row.map((val, j) => Math.random() < 0.5 ? val : partner.weights3[i][j])
        );

        child.bias1 = this.bias1.map((val, i) => Math.random() < 0.5 ? val : partner.bias1[i]);
        child.bias2 = this.bias2.map((val, i) => Math.random() < 0.5 ? val : partner.bias2[i]);
        child.bias3 = this.bias3.map((val, i) => Math.random() < 0.5 ? val : partner.bias3[i]);

        return child;
    }

    clone() {
        return new NeuralNetwork(
            this.inputNodes,
            this.hiddenNodes,
            this.outputNodes,
            { w1: this.weights1, w2: this.weights2, w3: this.weights3 },
            { b1: this.bias1, b2: this.bias2, b3: this.bias3 }
        );
    }
}

export class GeneticEvolution {
    /**
     * @param {number} populationSize 
     * @param {number} mutationRate 
     * @param {number} mutationStrength 
     */
    constructor(populationSize = 50, mutationRate = 0.10, mutationStrength = 0.20) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.mutationStrength = mutationStrength;

        this.generation = 1;
        this.currentAgentIndex = 0;
        this.generationBestFitness = 0;

        // Initialize population
        this.population = Array.from({ length: this.populationSize }, () => ({
            brain: new NeuralNetwork(6, 8, 3),
            fitness: 0
        }));
    }

    getCurrentAgent() {
        return this.population[this.currentAgentIndex];
    }

    /**
     * Registers fitness score for the current agent.
     * @param {number} score 
     * @returns {boolean} True if a new generation started.
     */
    registerScore(score) {
        this.population[this.currentAgentIndex].fitness = score;
        if (score > this.generationBestFitness) {
            this.generationBestFitness = score;
        }

        this.currentAgentIndex++;

        if (this.currentAgentIndex >= this.populationSize) {
            this.evolve();
            this.currentAgentIndex = 0;
            return true;
        }
        return false;
    }

    /**
     * Selects a candidate using Tournament Selection (k = tournamentSize).
     * Maintains healthy selection pressure while preserving population diversity.
     * @param {Array<Object>} pop 
     * @param {number} tournamentSize 
     * @returns {NeuralNetwork}
     */
    tournamentSelect(pop, tournamentSize = 3) {
        let best = null;
        for (let i = 0; i < tournamentSize; i++) {
            const randomIndex = Math.floor(Math.random() * pop.length);
            const candidate = pop[randomIndex];
            if (!best || candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        return best.brain;
    }

    /**
     * Breeds the next generation of agents.
     */
    evolve() {
        // Sort population descending by fitness
        this.population.sort((a, b) => b.fitness - a.fitness);

        // Keep top 10% as elites (minimum 2)
        const elitesCount = Math.max(2, Math.floor(this.populationSize * 0.10));
        const elites = this.population.slice(0, elitesCount).map(p => p.brain.clone());

        const newPopulation = [];

        // 1. Add Elites directly without mutation
        for (let i = 0; i < elitesCount; i++) {
            newPopulation.push({ brain: elites[i].clone(), fitness: 0 });
        }

        // 2. Breed remaining population via Tournament Selection + Crossover + Gaussian Mutation
        for (let i = elitesCount; i < this.populationSize; i++) {
            const parentA = this.tournamentSelect(this.population, 3);
            const parentB = this.tournamentSelect(this.population, 3);

            const childBrain = parentA.crossover(parentB);
            childBrain.mutate(this.mutationRate, this.mutationStrength);

            newPopulation.push({ brain: childBrain, fitness: 0 });
        }

        this.population = newPopulation;
        this.generationBestFitness = 0;
        this.generation++;
    }

    exportBestBrain() {
        const sorted = [...this.population].sort((a, b) => b.fitness - a.fitness);
        const best = sorted[0];
        return JSON.stringify({
            weights1: best.brain.weights1,
            weights2: best.brain.weights2,
            weights3: best.brain.weights3,
            bias1: best.brain.bias1,
            bias2: best.bias2,
            bias3: best.bias3
        });
    }
}
