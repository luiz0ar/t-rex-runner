/**
 * Q-Learning Policy implementation.
 * Manages the Q-table, action selection (epsilon-greedy), and value updates.
 */
export class QLearningPolicy {
    /**
     * @param {Object} config
     * @param {number} config.learningRate - Learning rate (alpha), default: 0.1
     * @param {number} config.discountFactor - Discount factor (gamma), default: 0.9
     * @param {number} config.epsilon - Exploration rate, default: 0.1
     * @param {Array<string>} config.actions - Available actions, default: ['NONE', 'JUMP', 'DUCK']
     */
    constructor(config = {}) {
        this.learningRate = config.learningRate !== undefined ? config.learningRate : 0.1;
        this.discountFactor = config.discountFactor !== undefined ? config.discountFactor : 0.9;
        this.epsilon = config.epsilon !== undefined ? config.epsilon : 0.1;
        this.actions = config.actions || ['NONE', 'JUMP', 'DUCK'];
        
        // Q-table: Map of { [stateKey]: { [action]: number } }
        this.qTable = {};
    }

    /**
     * Ensures a state exists in the Q-table with default values (0.0).
     * @private
     * @param {string} state - The state key
     */
    _ensureState(state) {
        if (!this.qTable[state]) {
            this.qTable[state] = {};
            for (const action of this.actions) {
                this.qTable[state][action] = 0.0;
            }
        }
    }

    /**
     * Selects an action based on Epsilon-Greedy policy.
     * @param {string} state - Discretized state key
     * @returns {string} The chosen action
     */
    selectAction(state) {
        this._ensureState(state);

        // Exploration: Choose a random action
        if (Math.random() < this.epsilon) {
            const randomIndex = Math.floor(Math.random() * this.actions.length);
            return this.actions[randomIndex];
        }

        // Exploitation: Choose the best action
        return this.getBestAction(state);
    }

    /**
     * Returns the action with the highest Q-value for a given state.
     * @param {string} state - Discretized state key
     * @returns {string} The best action
     */
    getBestAction(state) {
        this._ensureState(state);

        const actions = this.qTable[state];
        let bestAction = this.actions[0];
        let maxVal = -Infinity;

        // Shuffle actions to break ties randomly
        const shuffledActions = [...this.actions].sort(() => Math.random() - 0.5);

        for (const action of shuffledActions) {
            const value = actions[action];
            if (value > maxVal) {
                maxVal = value;
                bestAction = action;
            }
        }

        return bestAction;
    }

    /**
     * Updates the Q-value for a state-action pair using the Bellman Equation.
     * Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
     * 
     * @param {string} state - The current discretized state
     * @param {string} action - The action taken
     * @param {number} reward - The reward received
     * @param {string} nextState - The resulting discretized state
     */
    update(state, action, reward, nextState) {
        this._ensureState(state);
        this._ensureState(nextState);

        const currentQ = this.qTable[state][action];
        
        // Find max Q value for the next state: max(Q(s', a'))
        const nextStateActions = this.qTable[nextState];
        const maxNextQ = Math.max(...Object.values(nextStateActions));

        // Temporal Difference target
        const tdTarget = reward + this.discountFactor * maxNextQ;
        
        // Update Q-value
        this.qTable[state][action] = currentQ + this.learningRate * (tdTarget - currentQ);
    }

    /**
     * Returns the size of the Q-table (number of explored states).
     * @returns {number}
     */
    getQTableSize() {
        return Object.keys(this.qTable).length;
    }

    /**
     * Exports the Q-table as a JSON string.
     * @returns {string}
     */
    exportTable() {
        return JSON.stringify(this.qTable);
    }

    /**
     * Imports a Q-table from a JSON object or string.
     * @param {Object|string} data
     */
    importTable(data) {
        if (typeof data === 'string') {
            this.qTable = JSON.parse(data);
        } else {
            this.qTable = { ...data };
        }
    }
}
