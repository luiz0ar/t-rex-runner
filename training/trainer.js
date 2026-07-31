import { GameAdapter } from '../infrastructure/game-adapter.js';
import { discretizeState } from '../domain/discretizer.js';
import { QLearningPolicy } from '../application/q-learning-policy.js';

class AITrainer {
    constructor() {
        this.adapter = new GameAdapter();
        this.policy = new QLearningPolicy({
            learningRate: 0.15,
            discountFactor: 0.90,
            epsilon: 0.2, // Start with some exploration
            actions: ['NONE', 'JUMP', 'DUCK']
        });

        // Training parameters
        this.epsilonDecay = 0.995;
        this.minEpsilon = 0.001;
        this.episodeCount = 0;
        this.highestScore = 0;
        this.scores = [];

        // Loop state
        this.lastState = null;
        this.lastAction = null;
        this.isTrainingActive = false;
        this.isReplayMode = false;

        // Load saved progress if exists
        this.loadProgress();

        // Initialize UI
        this.initUI();
    }

    start() {
        this.isTrainingActive = true;
        this.lastState = null;
        this.lastAction = null;
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
            // Auto start if training is active
            this.adapter.reset();
            this.lastState = null;
            this.lastAction = null;
            requestAnimationFrame(() => this.loop());
            return;
        }

        const rawState = this.adapter.readRawState();
        if (!rawState) {
            requestAnimationFrame(() => this.loop());
            return;
        }

        // Update real-time stats on UI
        this.updateLiveStats(rawState.score);

        const currentState = discretizeState(rawState);

        // Update Q-value for the previous step if it exists (only if not in replay mode)
        if (!this.isReplayMode && this.lastState && this.lastAction) {
            // Base reward for surviving a frame
            let reward = 1.0;

            // Reward Refinement: Penalize unnecessary jumps.
            // If the T-Rex was on the ground and decided to JUMP when the obstacle was far away or non-existent:
            if (this.lastAction === 'JUMP' &&
                this.lastState.endsWith('_on_ground') &&
                (this.lastState.startsWith('far_') || this.lastState === 'NO_OBSTACLE')) {
                reward = -100.0; // Apply a penalty to prevent jumping without danger
            }

            this.policy.update(this.lastState, this.lastAction, reward, currentState);
        }

        // Choose next action. Replay mode uses the best learned action (exploitation only)
        const action = this.isReplayMode
            ? this.policy.getBestAction(currentState)
            : this.policy.selectAction(currentState);

        this.adapter.applyAction(action);

        this.lastState = currentState;
        this.lastAction = action;

        requestAnimationFrame(() => this.loop());
    }

    handleGameOver() {
        if (!this.isReplayMode && this.lastState && this.lastAction) {
            // Negative reward for colliding
            const reward = -1000.0;
            this.policy.update(this.lastState, this.lastAction, reward, 'CRASHED');
        }

        // Get final score to record
        const rawState = this.adapter.readRawState();
        const finalScore = rawState ? rawState.score : 0;

        // Save score to history
        this.scores.push(finalScore);

        if (finalScore > this.highestScore) {
            this.highestScore = finalScore;
        }

        this.episodeCount++;

        // Decay exploration rate (only when training)
        if (!this.isReplayMode) {
            this.policy.epsilon = Math.max(this.minEpsilon, this.policy.epsilon * this.epsilonDecay);

            // Auto save periodically
            if (this.episodeCount % 5 === 0) {
                this.saveProgress();
            }
        }

        this.lastState = null;
        this.lastAction = null;

        this.updateUI();

        // Delay reset slightly so the user can see game over state
        setTimeout(() => {
            if (this.isTrainingActive && this.adapter.isGameOver()) {
                this.adapter.reset();
            }
        }, 800);
    }

    saveProgress() {
        localStorage.setItem('trex_qtable', this.policy.exportTable());
        localStorage.setItem('trex_episodes', this.episodeCount.toString());
        localStorage.setItem('trex_highscore', this.highestScore.toString());
        localStorage.setItem('trex_epsilon', this.policy.epsilon.toString());
        localStorage.setItem('trex_scores', JSON.stringify(this.scores));
    }

    loadProgress() {
        const qtable = localStorage.getItem('trex_qtable');
        const episodes = localStorage.getItem('trex_episodes');
        const highscore = localStorage.getItem('trex_highscore');
        const epsilon = localStorage.getItem('trex_epsilon');
        const savedScores = localStorage.getItem('trex_scores');

        if (qtable) this.policy.importTable(qtable);
        if (episodes) this.episodeCount = parseInt(episodes, 10);
        if (highscore) this.highestScore = parseInt(highscore, 10);
        if (epsilon) this.policy.epsilon = parseFloat(epsilon);
        if (savedScores) this.scores = JSON.parse(savedScores);
    }

    resetBrain() {
        if (confirm('Are you sure you want to reset all AI learning?')) {
            localStorage.removeItem('trex_qtable');
            localStorage.removeItem('trex_episodes');
            localStorage.removeItem('trex_highscore');
            localStorage.removeItem('trex_epsilon');
            localStorage.removeItem('trex_scores');

            this.policy.qTable = {};
            this.policy.epsilon = 0.2;
            this.episodeCount = 0;
            this.highestScore = 0;
            this.scores = [];
            this.updateUI();
        }
    }

    initUI() {
        // Create container
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
            width: 250px;
        `;

        container.innerHTML = `
            <div><b>T-Rex AI (Q-Learning)</b></div>
            <div>Status: <span id="ai-status-badge">INACTIVE</span></div>
            <div>Mode: <span id="ui-mode-badge">TRAIN</span></div>
            <hr/>
            <div>Episodes: <span id="ui-episodes">0</span></div>
            <div>Q-Table size: <span id="ui-states">0</span></div>
            <div>Epsilon: <span id="ui-epsilon">0.0%</span></div>
            <div>High Score: <span id="ui-highscore">0</span></div>
            <div>Score: <span id="ui-current-score">0</span></div>
            <hr/>
            <div id="ui-chart" style="line-height: 1.2; font-size: 11px;"></div>
            <hr/>
            <button id="btn-toggle-ai">Start Training</button>
            <button id="btn-toggle-mode">Toggle Mode</button>
            <button id="btn-reset-ai">Reset</button>
            <button id="btn-save-ai">Export JSON</button>
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

        document.getElementById('btn-toggle-mode').addEventListener('click', () => {
            this.isReplayMode = !this.isReplayMode;
            this.updateUI();
        });

        document.getElementById('btn-reset-ai').addEventListener('click', () => this.resetBrain());
        document.getElementById('btn-save-ai').addEventListener('click', () => {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(this.policy.exportTable());
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", `qtable_episode_${this.episodeCount}.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        });

        this.updateUI();
    }

    updateUI() {
        const statusBadge = document.getElementById('ai-status-badge');
        const btnToggle = document.getElementById('btn-toggle-ai');
        const modeBadge = document.getElementById('ui-mode-badge');
        const btnToggleMode = document.getElementById('btn-toggle-mode');

        if (this.isTrainingActive) {
            statusBadge.innerText = 'ACTIVE';
            statusBadge.style.color = 'green';
            btnToggle.innerText = 'Pause';
        } else {
            statusBadge.innerText = 'INACTIVE';
            statusBadge.style.color = 'red';
            btnToggle.innerText = 'Start';
        }

        if (this.isReplayMode) {
            modeBadge.innerText = 'REPLAY (Exploit)';
            modeBadge.style.color = 'blue';
            btnToggleMode.innerText = 'Switch to Train';
        } else {
            modeBadge.innerText = 'TRAIN (Exploring)';
            modeBadge.style.color = 'darkorange';
            btnToggleMode.innerText = 'Switch to Replay';
        }

        document.getElementById('ui-episodes').innerText = this.episodeCount;
        document.getElementById('ui-states').innerText = this.policy.getQTableSize();
        document.getElementById('ui-epsilon').innerText = `${(this.policy.epsilon * 100).toFixed(1)}%`;
        document.getElementById('ui-highscore').innerText = this.highestScore;

        this.updateChart();
    }

    updateLiveStats(score) {
        document.getElementById('ui-current-score').innerText = score;
        document.getElementById('ui-states').innerText = this.policy.getQTableSize();
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
        const recentAverages = averages.slice(-5); // Show last 5 blocks
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
