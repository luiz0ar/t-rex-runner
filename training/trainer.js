import { GameAdapter } from '../infrastructure/game-adapter.js';
import { discretizeState } from '../domain/discretizer.js';
import { QLearningPolicy } from '../application/q-learning-policy.js';
import { GeneticEvolution } from '../application/genetic-policy.js';

class AITrainer {
    constructor() {
        this.adapter = new GameAdapter();
        
        // Algorithm mode: 'q-learning' or 'genetic'
        this.algorithm = 'q-learning';

        // Q-Learning policy
        this.policy = new QLearningPolicy({
            learningRate: 0.15,
            discountFactor: 0.90,
            epsilon: 0.5,
            actions: ['NONE', 'JUMP', 'DUCK']
        });

        // Genetic / Neuroevolution policy
        this.genetic = new GeneticEvolution(10, 0.15); // 10 agents per generation, 15% mutation rate

        // Training parameters
        this.epsilonDecay = 0.998;
        this.minEpsilon = 0.01;
        this.episodeCount = 0;
        this.highestScore = 0;
        this.scores = [];

        // Loop state
        this.lastState = null;
        this.lastAction = null;
        this.lastObstacleRef = null;
        this.isTrainingActive = false;
        this.isReplayMode = false;
        this.accumulatedReward = 0;

        // Load saved progress if exists
        this.loadProgress();

        // Initialize UI
        this.initUI();
    }

    start() {
        this.isTrainingActive = true;
        this.lastState = null;
        this.lastAction = null;
        this.lastObstacleRef = null;
        this.accumulatedReward = 0;
        this.updateUI();
        this.loop();
    }

    stop() {
        this.isTrainingActive = false;
        this.updateUI();
    }

    loop() {
        if (!this.isTrainingActive) return;

        // Check if game is crashed / gameover
        if (this.adapter.isGameOver()) {
            this.handleGameOver();
            requestAnimationFrame(() => this.loop());
            return;
        }

        // Check if game is not started or ready
        if (!this.adapter.isPlaying()) {
            this.adapter.reset();
            this.lastState = null;
            this.lastAction = null;
            this.lastObstacleRef = null;
            this.accumulatedReward = 0;
            requestAnimationFrame(() => this.loop());
            return;
        }

        const rawState = this.adapter.readRawState();
        if (!rawState) {
            requestAnimationFrame(() => this.loop());
            return;
        }

        // Update real-time score display
        this.updateLiveStats(rawState.score);

        if (this.algorithm === 'q-learning') {
            this.runQLearningStep(rawState);
        } else if (this.algorithm === 'genetic') {
            this.runGeneticStep(rawState);
        }

        requestAnimationFrame(() => this.loop());
    }

    runQLearningStep(rawState) {
        const currentState = discretizeState(rawState);

        // Frame reward calculations
        let frameReward = 0.1;
        
        if (this.lastObstacleRef && this.lastObstacleRef !== rawState.nextObstacle) {
            frameReward = 20.0;
        }
        this.lastObstacleRef = rawState.nextObstacle;
        this.accumulatedReward += frameReward;

        if (currentState !== this.lastState) {
            if (!this.isReplayMode && this.lastState && this.lastAction) {
                if (this.lastAction === 'JUMP' && 
                    this.lastState.endsWith('_on_ground') && 
                    (this.lastState.startsWith('far_') || this.lastState === 'NO_OBSTACLE')) {
                    this.accumulatedReward -= 5.0;
                }
                this.policy.update(this.lastState, this.lastAction, this.accumulatedReward, currentState);
            }

            this.accumulatedReward = 0;

            const action = this.isReplayMode
                ? this.policy.getBestAction(currentState)
                : this.policy.selectAction(currentState);
            
            this.adapter.applyAction(action);

            this.lastState = currentState;
            this.lastAction = action;
        } else if (this.lastAction) {
            this.adapter.applyAction(this.lastAction);
        }
    }

    runGeneticStep(rawState) {
        const agent = this.genetic.getCurrentAgent();
        const tRex = rawState.tRex;
        const obs = rawState.nextObstacle;

        // Neural Network inputs (normalized 0.0 to 1.0)
        const inputs = [
            obs ? Math.min(1.0, (obs.x - tRex.x) / 600) : 1.0, // Distance
            obs ? Math.min(1.0, obs.width / 100) : 0,         // Width
            obs ? Math.min(1.0, obs.height / 100) : 0,        // Height
            obs ? Math.min(1.0, obs.y / 150) : 0,             // Obstacle Y pos
            Math.min(1.0, rawState.gameSpeed / 20),           // Game speed
            Math.min(1.0, tRex.y / 150)                       // T-Rex Y position (air status)
        ];

        const outputs = agent.brain.predict(inputs);

        // Map output activations to action
        // Output indexes: 0 = NONE, 1 = JUMP, 2 = DUCK
        const maxIndex = outputs.indexOf(Math.max(...outputs));
        const actions = ['NONE', 'JUMP', 'DUCK'];
        const action = actions[maxIndex];

        this.adapter.applyAction(action);
        this.lastAction = action;
    }

    handleGameOver() {
        const rawState = this.adapter.readRawState();
        const finalScore = rawState ? rawState.score : 0;

        if (this.algorithm === 'q-learning') {
            if (!this.isReplayMode && this.lastState && this.lastAction) {
                const reward = this.accumulatedReward - 1000.0;
                this.policy.update(this.lastState, this.lastAction, reward, 'CRASHED');
            }

            this.scores.push(finalScore);

            if (finalScore > this.highestScore) {
                this.highestScore = finalScore;
            }

            this.episodeCount++;

            if (!this.isReplayMode) {
                this.policy.epsilon = Math.max(this.minEpsilon, this.policy.epsilon * this.epsilonDecay);
                if (this.episodeCount % 5 === 0) {
                    this.saveProgress();
                }
            }
        } else if (this.algorithm === 'genetic') {
            // Register fitness score for current agent and check if generation is complete
            const isNewGen = this.genetic.registerScore(finalScore);

            this.scores.push(finalScore);

            if (finalScore > this.highestScore) {
                this.highestScore = finalScore;
            }

            if (isNewGen) {
                this.saveProgress();
            }
        }

        this.lastState = null;
        this.lastAction = null;
        this.lastObstacleRef = null;
        this.accumulatedReward = 0;

        this.updateUI();

        // Delay reset slightly to let the user see the crash
        setTimeout(() => {
            if (this.isTrainingActive && this.adapter.isGameOver()) {
                this.adapter.reset();
            }
        }, 800);
    }

    saveProgress() {
        localStorage.setItem('trex_algo', this.algorithm);
        localStorage.setItem('trex_highscore', this.highestScore.toString());
        localStorage.setItem('trex_scores', JSON.stringify(this.scores));

        // Save Q-learning parameters
        localStorage.setItem('trex_qtable', this.policy.exportTable());
        localStorage.setItem('trex_episodes', this.episodeCount.toString());
        localStorage.setItem('trex_epsilon', this.policy.epsilon.toString());

        // Save Genetic parameters
        localStorage.setItem('trex_generation', this.genetic.generation.toString());
        localStorage.setItem('trex_agent_idx', this.genetic.currentAgentIndex.toString());
        localStorage.setItem('trex_population', JSON.stringify(this.genetic.population.map(p => ({
            weights1: p.brain.weights1,
            weights2: p.brain.weights2,
            bias1: p.brain.bias1,
            bias2: p.brain.bias2,
            fitness: p.fitness
        }))));
    }

    loadProgress() {
        const algo = localStorage.getItem('trex_algo');
        const highscore = localStorage.getItem('trex_highscore');
        const savedScores = localStorage.getItem('trex_scores');

        if (algo) this.algorithm = algo;
        if (highscore) this.highestScore = parseInt(highscore, 10);
        if (savedScores) this.scores = JSON.parse(savedScores);

        // Load Q-learning properties
        const qtable = localStorage.getItem('trex_qtable');
        const episodes = localStorage.getItem('trex_episodes');
        const epsilon = localStorage.getItem('trex_epsilon');

        if (qtable) this.policy.importTable(qtable);
        if (episodes) this.episodeCount = parseInt(episodes, 10);
        if (epsilon) this.policy.epsilon = parseFloat(epsilon);

        // Load Genetic properties
        const gen = localStorage.getItem('trex_generation');
        const agentIdx = localStorage.getItem('trex_agent_idx');
        const popData = localStorage.getItem('trex_population');

        if (gen) this.genetic.generation = parseInt(gen, 10);
        if (agentIdx) this.genetic.currentAgentIndex = parseInt(agentIdx, 10);
        if (popData) {
            const list = JSON.parse(popData);
            this.genetic.population = list.map(item => ({
                brain: new NeuralNetwork(6, 6, 3, 
                    { w1: item.weights1, w2: item.weights2 }, 
                    { b1: item.bias1, b2: item.bias2 }
                ),
                fitness: item.fitness
            }));
        }
    }

    resetBrain() {
        if (confirm('Are you sure you want to reset all AI progress?')) {
            localStorage.clear();
            
            // Re-instantiate policies
            this.policy = new QLearningPolicy({
                learningRate: 0.15,
                discountFactor: 0.90,
                epsilon: 0.5,
                actions: ['NONE', 'JUMP', 'DUCK']
            });
            this.genetic = new GeneticEvolution(10, 0.15);

            this.episodeCount = 0;
            this.highestScore = 0;
            this.scores = [];
            this.updateUI();
        }
    }

    initUI() {
        const container = document.createElement('div');
        container.id = 'ai-dashboard';
        container.style.cssText = `
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: #eee;
            border: 2px solid #555;
            padding: 10px;
            color: #000;
            font-family: monospace;
            font-size: 12px;
            z-index: 9999;
            width: 255px;
        `;

        container.innerHTML = `
            <div><b>T-Rex AI Brain</b></div>
            <div>Status: <span id="ai-status-badge">INACTIVE</span></div>
            <div>Algo: <span id="ui-algo-badge" style="font-weight: bold;">Q-LEARNING</span></div>
            <div id="ui-mode-row">Mode: <span id="ui-mode-badge">TRAIN</span></div>
            <hr/>
            <div id="ui-stats-container">
                <!-- Dynamically populated stats -->
            </div>
            <hr/>
            <div id="ui-chart" style="line-height: 1.2; font-size: 11px;"></div>
            <hr/>
            <button id="btn-toggle-ai">Start</button>
            <button id="btn-toggle-algo">Switch Algorithm</button>
            <button id="btn-toggle-mode">Toggle Mode</button>
            <button id="btn-reset-ai">Reset</button>
            <button id="btn-save-ai">Export Best</button>
        `;

        document.body.appendChild(container);

        // Bind events
        document.getElementById('btn-toggle-ai').addEventListener('click', () => {
            if (this.isTrainingActive) {
                this.stop();
            } else {
                this.start();
            }
        });

        document.getElementById('btn-toggle-algo').addEventListener('click', () => {
            this.stop();
            this.algorithm = this.algorithm === 'q-learning' ? 'genetic' : 'q-learning';
            this.saveProgress();
            this.updateUI();
        });

        document.getElementById('btn-toggle-mode').addEventListener('click', () => {
            this.isReplayMode = !this.isReplayMode;
            this.updateUI();
        });

        document.getElementById('btn-reset-ai').addEventListener('click', () => this.resetBrain());
        
        document.getElementById('btn-save-ai').addEventListener('click', () => {
            let dataStr;
            let filename;
            if (this.algorithm === 'q-learning') {
                dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(this.policy.exportTable());
                filename = `qtable_ep_${this.episodeCount}.json`;
            } else {
                dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(this.genetic.exportBestBrain());
                filename = `neuro_best_gen_${this.genetic.generation}.json`;
            }
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", filename);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        });

        this.updateUI();
    }

    updateUI() {
        const statusBadge = document.getElementById('ai-status-badge');
        const btnToggle = document.getElementById('btn-toggle-ai');
        const algoBadge = document.getElementById('ui-algo-badge');
        const modeBadge = document.getElementById('ui-mode-badge');
        const btnToggleMode = document.getElementById('btn-toggle-mode');
        const uiModeRow = document.getElementById('ui-mode-row');
        const statsContainer = document.getElementById('ui-stats-container');

        if (this.isTrainingActive) {
            statusBadge.innerText = 'ACTIVE';
            statusBadge.style.color = 'green';
            btnToggle.innerText = 'Pause';
        } else {
            statusBadge.innerText = 'INACTIVE';
            statusBadge.style.color = 'red';
            btnToggle.innerText = 'Start';
        }

        if (this.algorithm === 'q-learning') {
            algoBadge.innerText = 'Q-LEARNING';
            algoBadge.style.color = 'purple';
            uiModeRow.style.display = 'block';
            btnToggleMode.style.display = 'inline-block';

            if (this.isReplayMode) {
                modeBadge.innerText = 'REPLAY (Exploit)';
                modeBadge.style.color = 'blue';
                btnToggleMode.innerText = 'Switch to Train';
            } else {
                modeBadge.innerText = 'TRAIN (Exploring)';
                modeBadge.style.color = 'darkorange';
                btnToggleMode.innerText = 'Switch to Replay';
            }

            statsContainer.innerHTML = `
                <div>Episodes: <span id="ui-episodes">${this.episodeCount}</span></div>
                <div>Q-Table size: <span id="ui-states">${this.policy.getQTableSize()}</span></div>
                <div>Epsilon: <span id="ui-epsilon">${(this.policy.epsilon * 100).toFixed(1)}%</span></div>
                <div>High Score: <span id="ui-highscore">${this.highestScore}</span></div>
                <div>Score: <span id="ui-current-score">0</span></div>
            `;
        } else {
            algoBadge.innerText = 'NEUROEVOLUTION';
            algoBadge.style.color = 'teal';
            uiModeRow.style.display = 'none';
            btnToggleMode.style.display = 'none';

            statsContainer.innerHTML = `
                <div>Generation: <span id="ui-generation">${this.genetic.generation}</span></div>
                <div>Dino Index: <span id="ui-agent-idx">${this.genetic.currentAgentIndex + 1}/${this.genetic.populationSize}</span></div>
                <div>Mutation: <span id="ui-mutation">${(this.genetic.mutationRate * 100).toFixed(0)}%</span></div>
                <div>High Score: <span id="ui-highscore">${this.highestScore}</span></div>
                <div>Score: <span id="ui-current-score">0</span></div>
            `;
        }

        this.updateChart();
    }

    updateLiveStats(score) {
        document.getElementById('ui-current-score').innerText = score;
        if (this.algorithm === 'q-learning') {
            document.getElementById('ui-states').innerText = this.policy.getQTableSize();
        }
    }

    updateChart() {
        const uiChart = document.getElementById('ui-chart');
        if (!uiChart) return;

        if (this.scores.length === 0) {
            uiChart.innerHTML = '<b>Evolution (avg of 10 eps):</b><br/>No scores yet.';
            return;
        }

        const blockSize = 10;
        const averages = [];
        for (let i = 0; i < this.scores.length; i += blockSize) {
            const chunk = this.scores.slice(i, i + blockSize);
            const avg = chunk.reduce((sum, s) => sum + s, 0) / chunk.length;
            averages.push({
                range: `${i + 1}-${i + chunk.length}`,
                avg: Math.round(avg)
            });
        }

        const maxBarLength = 12;
        const maxAvg = Math.max(...averages.map(a => a.avg), 100);

        let chartHtml = '<b>Evolution (avg of 10 eps):</b><br/>';
        const recentAverages = averages.slice(-5);
        for (const item of recentAverages) {
            const barCount = Math.max(1, Math.min(maxBarLength, Math.round((item.avg / maxAvg) * maxBarLength)));
            const barStr = '█'.repeat(barCount) + '░'.repeat(maxBarLength - barCount);
            chartHtml += `${String(item.range).padEnd(6)}: ${barStr} (${item.avg})<br/>`;
        }

        uiChart.innerHTML = chartHtml;
    }
}

// Automatically instantiate and attach to global scope
document.addEventListener('DOMContentLoaded', () => {
    // Wait for Chrome runner to load
    setTimeout(() => {
        window.aiTrainer = new AITrainer();
    }, 1000);
});
