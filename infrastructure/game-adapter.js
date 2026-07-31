/**
 * Adapts the internal state and controls of Chrome's T-Rex Runner game.
 * Binds the DOM game instance to the clean architecture application layer.
 */
export class GameAdapter {
    constructor() {
        this.runner = null;
    }

    /**
     * Ensures the Runner instance is captured.
     * @private
     * @returns {boolean} True if ready.
     */
    _ensureRunner() {
        if (!this.runner && window.Runner && window.Runner.instance_) {
            this.runner = window.Runner.instance_;
        }
        return !!this.runner;
    }

    /**
     * Reads the current raw game state.
     * @returns {Object|null} Raw state or null if the game is not ready.
     */
    readRawState() {
        if (!this._ensureRunner()) return null;

        const tRex = this.runner.tRex;
        const obstacles = this.runner.horizon ? this.runner.horizon.obstacles : [];
        
        // Find the next obstacle that the T-Rex has not yet cleared.
        // T-Rex front boundary is roughly xPos (50) + width.
        const tRexFront = tRex.xPos;
        const nextObstacle = obstacles.find(
            obs => obs.xPos + obs.width > tRexFront
        );

        return {
            gameSpeed: this.runner.currentSpeed,
            crashed: this.runner.crashed,
            playing: this.runner.playing,
            tRex: {
                x: tRex.xPos,
                y: tRex.yPos,
                jumping: tRex.jumping,
                ducking: tRex.ducking,
                groundY: tRex.groundYPos
            },
            nextObstacle: nextObstacle ? {
                x: nextObstacle.xPos,
                y: nextObstacle.yPos,
                width: nextObstacle.width,
                height: nextObstacle.typeConfig.height,
                type: nextObstacle.typeConfig.type
            } : null,
            score: this.runner.distanceMeter ? Math.round(this.runner.distanceMeter.getActualDistance(this.runner.distanceRan)) : 0
        };
    }

    /**
     * Checks if the game is over.
     * @returns {boolean}
     */
    isGameOver() {
        if (!this._ensureRunner()) return false;
        return this.runner.crashed;
    }

    /**
     * Checks if the game is actively running.
     * @returns {boolean}
     */
    isPlaying() {
        if (!this._ensureRunner()) return false;
        return this.runner.playing && !this.runner.crashed;
    }

    /**
     * Resets/Restarts the game.
     */
    reset() {
        if (!this._ensureRunner()) return;

        if (this.runner.crashed) {
            this.runner.restart();
        } else if (!this.runner.playing) {
            // Start the game by simulating Space keydown
            this.triggerKeyDown(32);
        }
    }

    /**
     * Applies a control action to the T-Rex.
     * Actions: 'JUMP', 'DUCK', 'NONE'
     * @param {string} action
     */
    applyAction(action) {
        if (!this._ensureRunner() || this.runner.crashed) return;

        const tRex = this.runner.tRex;

        if (action === 'JUMP') {
            if (tRex.ducking) {
                this.triggerKeyUp(40); // Release ArrowDown
            }
            this.triggerKeyDown(32); // Press Space to Jump
        } else if (action === 'DUCK') {
            this.triggerKeyDown(40); // Press ArrowDown to Duck/Speed Drop
        } else if (action === 'NONE') {
            if (tRex.ducking) {
                this.triggerKeyUp(40); // Release Ducking if it was active
            }
            // Release Jump key
            this.triggerKeyUp(32);
        }
    }

    /**
     * Dispatches keydown event on the document.
     * @param {number} keyCode
     */
    triggerKeyDown(keyCode) {
        const event = new KeyboardEvent('keydown', {
            keyCode: keyCode,
            which: keyCode,
            bubbles: true,
            cancelable: true
        });
        document.dispatchEvent(event);
    }

    /**
     * Dispatches keyup event on the document.
     * @param {number} keyCode
     */
    triggerKeyUp(keyCode) {
        const event = new KeyboardEvent('keyup', {
            keyCode: keyCode,
            which: keyCode,
            bubbles: true,
            cancelable: true
        });
        document.dispatchEvent(event);
    }
}
