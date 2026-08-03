import { GameAdapter } from '../infrastructure/game-adapter.js';
import { GeneticEvolution } from '../application/genetic-policy.js';

class AITrainer {
    constructor() {
        this.adapter = new GameAdapter();

        // Neuroevolution policy: 50 population, 10% mutation rate, 0.20 noise strength
        this.genetic = new GeneticEvolution(50, 0.10, 0.20);

        // Training stats
        this.highestScore = 0;
        this.scores = [];
        this.generationAverages = [];

        // Loop state
        this.lastObstacleX = null;
        this.isTrainingActive = false;
        this.currentAgentFitness = 0;

        // Load saved progress if exists
        this.loadProgress();

        // Initialize UI
        this.initUI();
    }

    start() {
        this.isTrainingActive = true;
        this.lastObstacleX = null;
        this.currentAgentFitness = 0;
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
            this.lastObstacleX = null;
            this.currentAgentFitness = 0;
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

        // Run continuous 60 FPS Neuroevolution step
        this.runGeneticStep(rawState);

        requestAnimationFrame(() => this.loop());
    }

    /**
     * Continuous 60 FPS evaluation of the Neural Network.
     * Receives raw game state and applies controls on every frame.
     * @param {Object} rawState 
     */
    runGeneticStep(rawState) {
        const { tRex, nextObstacle, gameSpeed } = rawState;
        const agent = this.genetic.getCurrentAgent();

        // Detect obstacle clearance by tracking obstacle X position
        const currentObsX = nextObstacle ? nextObstacle.x : null;
        let clearedObstacle = false;
        if (this.lastObstacleX !== null) {
            if (currentObsX === null || currentObsX > this.lastObstacleX + 40) {
                clearedObstacle = true;
            }
        }
        this.lastObstacleX = currentObsX;

        // Fitness accumulation: continuous survival reward + obstacle clearance bonus
        this.currentAgentFitness += 0.1;
        if (clearedObstacle) {
            this.currentAgentFitness += 25.0; // Bonus for successfully passing an obstacle
        }

        // Calculate smooth continuous inputs normalized between [0, 1]
        const tRexFront = tRex.x + 44;
        const obsDistance = nextObstacle ? Math.max(0, nextObstacle.x - tRexFront) : 800;
        
        // timeToCollision: 0.0 = collision point, 1.0 = safe distance (speed-adjusted)
        const timeToCollision = nextObstacle ? Math.max(0.0, Math.min(1.0, obsDistance / (gameSpeed * 25))) : 1.0;
        
        // obstacleY: height of obstacle above ground (0.0 = low cactus/bird, up to 1.0 = high bird)
        const obstacleY = nextObstacle ? Math.max(0.0, Math.min(1.0, (tRex.groundY - nextObstacle.y) / 100)) : 0.0;
        
        // tRexY: height of T-Rex above ground (0.0 = ground level, 1.0 = top of jump)
        const tRexY = Math.max(0.0, Math.min(1.0, (tRex.groundY - tRex.y) / 100));

        const inputs = [
            timeToCollision,                                       // Input 0: Time to collision
            nextObstacle ? Math.min(1.0, nextObstacle.width / 100) : 0.0, // Input 1: Obstacle width
            nextObstacle ? Math.min(1.0, nextObstacle.height / 100) : 0.0,// Input 2: Obstacle height
            obstacleY,                                             // Input 3: Obstacle Y position
            Math.min(1.0, gameSpeed / 25),                         // Input 4: Game speed
            tRexY                                                  // Input 5: T-Rex Y position
        ];

        // Feed forward prediction (Outputs: 0 = NONE, 1 = JUMP, 2 = DUCK)
        const outputs = agent.brain.predict(inputs);
        const maxIndex = outputs.indexOf(Math.max(...outputs));
        const actions = ['NONE', 'JUMP', 'DUCK'];
        const action = actions[maxIndex];

        // Small penalty for unnecessary jump spam when obstacles are far away
        if (action === 'JUMP' && tRexY === 0 && timeToCollision > 0.8) {
            this.currentAgentFitness -= 0.5;
        }

        // Apply control action to game
        this.adapter.applyAction(action);
    }

    handleGameOver() {
        const rawState = this.adapter.readRawState();
        const finalScore = rawState ? rawState.score : 0;

        // Calculate final fitness score combining distance ran and obstacle bonuses
        const finalFitness = Math.max(0.1, this.currentAgentFitness + finalScore * 1.5);

        // Register fitness and check if generation completed
        const isNewGen = this.genetic.registerScore(finalFitness);

        this.scores.push(finalScore);
        if (finalScore > this.highestScore) {
            this.highestScore = finalScore;
        }

        if (isNewGen) {
            // Track generation average score
            const genChunk = this.scores.slice(-this.genetic.populationSize);
            const genAvg = Math.round(genChunk.reduce((a, b) => a + b, 0) / genChunk.length);
            this.generationAverages.push(genAvg);

            this.saveProgress();
        }

        this.lastObstacleX = null;
        this.currentAgentFitness = 0;

        this.updateUI();

        // Restart game after short delay
        setTimeout(() => {
            if (this.isTrainingActive && this.adapter.isGameOver()) {
                this.adapter.reset();
            }
        }, 500);
    }

    saveProgress() {
        localStorage.setItem('trex_highscore', this.highestScore.toString());
        localStorage.setItem('trex_scores', JSON.stringify(this.scores));
        localStorage.setItem('trex_gen_avgs', JSON.stringify(this.generationAverages));

        // Save Genetic parameters
        localStorage.setItem('trex_generation', this.genetic.generation.toString());
        localStorage.setItem('trex_agent_idx', this.genetic.currentAgentIndex.toString());
        localStorage.setItem('trex_population', JSON.stringify(this.genetic.population.map(p => ({
            weights1: p.brain.weights1,
            weights2: p.brain.weights2,
            weights3: p.brain.weights3,
            bias1: p.bias1,
            bias2: p.bias2,
            bias3: p.bias3,
            fitness: p.fitness
        }))));
    }

    loadProgress() {
        const highscore = localStorage.getItem('trex_highscore');
        const savedScores = localStorage.getItem('trex_scores');
        const genAvgs = localStorage.getItem('trex_gen_avgs');

        if (highscore) this.highestScore = parseInt(highscore, 10);
        if (savedScores) this.scores = JSON.parse(savedScores);
        if (genAvgs) this.generationAverages = JSON.parse(genAvgs);

        // Load Genetic properties
        const gen = localStorage.getItem('trex_generation');
        const agentIdx = localStorage.getItem('trex_agent_idx');
        const popData = localStorage.getItem('trex_population');

        if (gen) this.genetic.generation = parseInt(gen, 10);
        if (agentIdx) this.genetic.currentAgentIndex = parseInt(agentIdx, 10);
        if (popData) {
            try {
                const list = JSON.parse(popData);
                this.genetic.population = list.map(item => ({
                    brain: new NeuralNetwork(6, 8, 3,
                        { w1: item.weights1, w2: item.weights2, w3: item.weights3 },
                        { b1: item.bias1, b2: item.bias2, b3: item.bias3 }
                    ),
                    fitness: item.fitness || 0
                }));
                this.genetic.populationSize = this.genetic.population.length;
            } catch (e) {
                console.warn("Could not load population, re-initializing", e);
            }
        }
    }

    resetBrain() {
        if (confirm('Deseja realmente reiniciar todo o progresso da Neuroevolução?')) {
            localStorage.clear();

            this.genetic = new GeneticEvolution(50, 0.10, 0.20);
            this.highestScore = 0;
            this.scores = [];
            this.generationAverages = [];
            this.currentAgentFitness = 0;
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
            <div>Algo: <span id="ui-algo-badge" style="font-weight: bold; color: teal;">NEUROEVOLUTION</span></div>
            <hr/>
            <div id="ui-stats-container">
                <!-- Dynamically populated stats -->
            </div>
            <hr/>
            <div id="ui-chart" style="line-height: 1.2; font-size: 11px;"></div>
            <hr/>
            <button id="btn-toggle-ai">Start</button>
            <button id="btn-reset-ai">Reset</button>
            <button id="btn-save-ai">Export Best</button>
        `;

        document.body.appendChild(container);

        document.getElementById('btn-toggle-ai').addEventListener('click', () => {
            if (this.isTrainingActive) {
                this.stop();
            } else {
                this.start();
            }
        });

        document.getElementById('btn-reset-ai').addEventListener('click', () => this.resetBrain());

        document.getElementById('btn-save-ai').addEventListener('click', () => {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(this.genetic.exportBestBrain());
            const filename = `neuro_best_gen_${this.genetic.generation}.json`;
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

        statsContainer.innerHTML = `
            <div>Generation: <span id="ui-generation">${this.genetic.generation}</span></div>
            <div>Dino Index: <span id="ui-agent-idx">${this.genetic.currentAgentIndex + 1}/${this.genetic.populationSize}</span></div>
            <div>Mutation: <span id="ui-mutation">${(this.genetic.mutationRate * 100).toFixed(0)}%</span></div>
            <div>High Score: <span id="ui-highscore">${this.highestScore}</span></div>
            <div>Score: <span id="ui-current-score">0</span></div>
        `;

        this.updateChart();
    }

    updateLiveStats(score) {
        const scoreElem = document.getElementById('ui-current-score');
        if (scoreElem) scoreElem.innerText = score;
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
    setTimeout(() => {
        window.aiTrainer = new AITrainer();
    }, 1000);
});
