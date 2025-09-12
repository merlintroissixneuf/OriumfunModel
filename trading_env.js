const tf = require('@tensorflow/tfjs-node');

const ACTIONS = { BUY: 0, SELL: 1, HOLD: 2 };

class TradingEnvironment {
  constructor(dataframe, initialBalance = 10.0, commission = 0.001) {
    this.df = dataframe;
    this.initialBalance = initialBalance;
    this.commission = commission;
    
    // The environment needs to provide sequences, so we define the length here.
    this.sequenceLength = 60; 
  }

  // Resets the environment to the beginning of the data for a new episode.
  reset() {
    this.balance = this.initialBalance;
    this.btcOwned = 0;
    this.portfolioValue = this.initialBalance;
    // We start at the first point where we have a full sequence.
    this.currentIndex = this.sequenceLength - 1; 
    
    return this._getCurrentState();
  }

  // Gets the current market data as a sequence (the "state").
  _getCurrentState() {
    if (this.currentIndex >= this.df.shape[0]) {
      return null; // End of data
    }
    // Return a 2D Tensor of shape [sequenceLength, num_features]
    return tf.tensor2d(
      this.df.iloc({
        rows: [`${this.currentIndex - this.sequenceLength + 1}:${this.currentIndex + 1}`],
        columns: ['0:'] // All columns
      }).values
    );
  }

  // The main function where the agent interacts with the environment.
  step(action) {
    this.currentIndex++;
    
    if (this.currentIndex >= this.df.shape[0]) {
      // We've run out of data
      return { state: null, reward: 0, done: true };
    }

    const currentPrice = this.df.iloc({ rows: [this.currentIndex], columns: ['Close'] }).values[0];
    let reward = 0;

    switch (action) {
      case ACTIONS.BUY:
        if (this.balance > 0) {
          const buyAmount = this.balance / currentPrice;
          this.btcOwned += buyAmount * (1 - this.commission); // Apply commission
          this.balance = 0;
        }
        break;
      case ACTIONS.SELL:
        if (this.btcOwned > 0) {
          this.balance += this.btcOwned * currentPrice * (1 - this.commission); // Apply commission
          this.btcOwned = 0;
        }
        break;
      case ACTIONS.HOLD:
        // Do nothing
        break;
    }
    
    const newPortfolioValue = this.balance + (this.btcOwned * currentPrice);
    
    // The reward is the change in portfolio value from the last step.
    reward = newPortfolioValue - this.portfolioValue;
    this.portfolioValue = newPortfolioValue;
    
    const nextState = this._getCurrentState();
    const done = this.currentIndex >= this.df.shape[0] - 1;

    return { state: nextState, reward, done };
  }
}

module.exports = { TradingEnvironment, ACTIONS };