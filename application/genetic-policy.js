/**
 * Simple Neural Network and Genetic Algorithm Policy for Neuroevolution.
 * Allows the T-Rex to learn using a feedforward neural network mutated over generations.
 */

export class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes, weights = null, biases = null) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        if (weights && biases) {
            this.weights1 = JSON.parse(JSON.stringify(weights.w1));
            this.weights2 = JSON.parse(JSON.stringify(weights.w2));
            this.bias1 = JSON.parse(JSON.stringify(biases.b1));
            this.bias2 = JSON.parse(JSON.stringify(biases.b2));
        } else {
            // Random initialization between -1.0 and 1.0
            this.weights1 = Array.from({ length: this.hiddenNodes }, () =>
                Array.from({ length: this.inputNodes }, () => Math.random() * 2 - 1)
            );
            this.weights2 = Array.from({ length: this.outputNodes }, () =>
                Array.from({ length: this.hiddenNodes }, () => Math.random() * 2 - 1)
            );
            this.bias1 = Array.from({ length: this.hiddenNodes }, () => Math.random() * 2 - 1);
            this.bias2 = Array.from({ length: this.outputNodes }, () => Math.random() * 2 - 1);
        }
    }

    /**
     * Sigmoid activation function.
     */
    _sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Feeds inputs forward through the network.
     * @param {Array<number>} inputs
     * @returns {Array<number>} Outputs
     */
    predict(inputs) {
        // Hidden layer activations
        const hidden = [];
        for (let i = 0; i < this.hiddenNodes; i++) {
            let sum = 0;
            for (let j = 0; j < this.inputNodes; j++) {
                sum += inputs[j] * this.weights1[i][j];
            }
            sum += this.bias1[i];
            hidden.push(this._sigmoid(sum));
        }

        // Output layer activations
        const outputs = [];
        for (let i = 0; i < this.outputNodes; i++) {
            let sum = 0;
            for (let j = 0; j < this.hiddenNodes; j++) {
                sum += hidden[j] * this.weights2[i][j];
            }
            sum += this.bias2[i];
            outputs.push(this._sigmoid(sum));
        }

        return outputs;
    }

    /**
     * Mutates weights and biases using a mutation rate.
     * @param {number} rate - Mutation rate (e.g. 0.1)
     */
    mutate(rate) {
        const mutateVal = (val) => {
            if (Math.random() < rate) {
                // Add a small Gaussian-like random change
                return val + (Math.random() * 2 - 1) * 0.5;
            }
            return val;
        };

        this.weights1 = this.weights1.map(row => row.map(mutateVal));
        this.weights2 = this.weights2.map(row => row.map(mutateVal));
        this.bias1 = this.bias1.map(mutateVal);
        this.bias2 = this.bias2.map(mutateVal);
    }

    clone() {
        return new NeuralNetwork(
            this.inputNodes,
            this.hiddenNodes,
            this.outputNodes,
            { w1: this.weights1, w2: this.weights2 },
            { b1: this.bias1, b2: this.bias2 }
        );
    }
}

export class GeneticEvolution {
    /**
     * @param {number} populationSize - Number of agents per generation
     * @param {number} mutationRate - Probability of mutation
     */
    constructor(populationSize = 10, mutationRate = 0.1) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        
        this.generation = 1;
        this.currentAgentIndex = 0;
        
        // Initialize population
        this.population = Array.from({ length: this.populationSize }, () => ({
            brain: new NeuralNetwork(6, 6, 3), // 6 inputs, 6 hidden nodes, 3 outputs (NONE, JUMP, DUCK)
            fitness: 0
        }));
    }

    /**
     * Get the currently active agent.
     */
    getCurrentAgent() {
        return this.population[this.currentAgentIndex];
    }

    /**
     * Register fitness score for the current agent and advance.
     * @param {number} score
     * @returns {boolean} True if a new generation has started.
     */
    registerScore(score) {
        this.population[this.currentAgentIndex].fitness = score;
        this.currentAgentIndex++;

        if (this.currentAgentIndex >= this.populationSize) {
            this.evolve();
            this.currentAgentIndex = 0;
            return true;
        }
        return false;
    }

    /**
     * Breed and mutate the next generation of agents based on fitness.
     */
    evolve() {
        // Sort population by fitness descending
        this.population.sort((a, b) => b.fitness - a.fitness);

        // Keep top 2 best performers (Elitism)
        const bestPerformers = [
            this.population[0].brain.clone(),
            this.population[1].brain.clone()
        ];

        const newPopulation = [];

        // Add elites directly
        newPopulation.push({ brain: bestPerformers[0].clone(), fitness: 0 });
        newPopulation.push({ brain: bestPerformers[1].clone(), fitness: 0 });

        // Generate offspring via replication and mutation
        for (let i = 2; i < this.populationSize; i++) {
            // Select parent (weighted towards best performers)
            const parentBrain = Math.random() < 0.7 ? bestPerformers[0] : bestPerformers[1];
            const childBrain = parentBrain.clone();
            childBrain.mutate(this.mutationRate);

            newPopulation.push({ brain: childBrain, fitness: 0 });
        }

        this.population = newPopulation;
        this.generation++;
    }

    /**
     * Export the best brain as a JSON string.
     */
    exportBestBrain() {
        const best = this.population.sort((a, b) => b.fitness - a.fitness)[0];
        return JSON.stringify({
            weights1: best.brain.weights1,
            weights2: best.brain.weights2,
            bias1: best.brain.bias1,
            bias2: best.brain.bias2
        });
    }
}
