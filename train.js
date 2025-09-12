// --- Imports ---
const df = require('danfojs-node');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { TradingEnvironment, ACTIONS } = require('./trading_env.js');

// --- Hyperparameters ---
const GAMMA = 0.95;             // Discount factor for future rewards
const EPSILON_START = 1.0;      // Starting exploration rate
const EPSILON_END = 0.01;       // Minimum exploration rate
const EPSILON_DECAY = 0.995;    // Rate at which exploration decreases
const LEARNING_RATE = 0.001;
const MEMORY_SIZE = 10000;      // Max number of experiences to store
const BATCH_SIZE = 64;          // Number of experiences to train on at once
const NUM_EPISODES = 50;        // How many times to run through the data

// --- The Agent's "Brain" ---
function createModel(sequenceLength, numFeatures) {
  const model = tf.sequential();
  model.add(tf.layers.reshape({ inputShape: [sequenceLength, numFeatures], targetShape: [sequenceLength, numFeatures, 1] }));
  model.add(tf.layers.conv1d({ kernelSize: 5, filters: 32, activation: 'relu' }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(tf.layers.lstm({ units: 50 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  // The output layer now has 3 units, one for each action (BUY, SELL, HOLD).
  // It predicts the Q-value (expected future reward) for each action.
  model.add(tf.layers.dense({ units: 3, activation: 'linear' })); 
  model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
  return model;
}

// --- The Agent's "Memory" ---
class ReplayMemory {
  constructor(capacity) {
    this.capacity = capacity;
    this.memory = [];
    this.position = 0;
  }

  push(state, action, reward, nextState, done) {
    if (this.memory.length < this.capacity) {
      this.memory.push(null);
    }
    this.memory[this.position] = { state, action, reward, nextState, done };
    this.position = (this.position + 1) % this.capacity;
  }

  sample(batchSize) {
    const samples = [];
    for (let i = 0; i < batchSize; i++) {
      const index = Math.floor(Math.random() * this.memory.length);
      samples.push(this.memory[index]);
    }
    return samples;
  }
}

// --- Main Training Function ---
async function runTraining() {
  console.log("--- Orium Reinforcement Learning Engine ---");

  // --- Step 1 & 2: Load Data and Engineer Features ---
  console.log("\nStep 1 & 2: Loading data and engineering features...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
  const closePrices = dataframe['Close'].values;
  const numPrices = closePrices.length;
  // (Feature engineering loops)
  let priceChangeValues = [0]; for (let i = 1; i < numPrices; i++) { const p = closePrices[i - 1], c = closePrices[i]; priceChangeValues.push(p === 0 ? 0 : ((c - p) / p) * 100); } dataframe.addColumn('Price_Change', priceChangeValues, { inplace: true });
  let sma10Values = []; for (let i = 0; i < numPrices; i++) { if (i < 9) { sma10Values.push(0); } else { let s = 0; for (let j = 0; j < 10; j++) { s += closePrices[i - j]; } sma10Values.push(s / 10); } } dataframe.addColumn('SMA_10', sma10Values, { inplace: true });
  let sma30Values = []; for (let i = 0; i < numPrices; i++) { if (i < 29) { sma30Values.push(0); } else { let s = 0; for (let j = 0; j < 30; j++) { s += closePrices[i - j]; } sma30Values.push(s / 30); } } dataframe.addColumn('SMA_30', sma30Values, { inplace: true });
  dataframe.dropNa({ inplace: true });
  
  // Normalize the data
  const featureCols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'SMA_10', 'SMA_30'];
  const scaler = new df.MinMaxScaler();
  scaler.fit(dataframe.loc({ columns: featureCols }));
  const scaledDf = scaler.transform(dataframe.loc({ columns: featureCols }));
  // We need to keep the original 'Close' price for reward calculation.
  scaledDf.addColumn('Close', dataframe['Close'], { inplace: true });

  // --- Initialize Environment and Agent ---
  const env = new TradingEnvironment(scaledDf);
  const sequenceLength = env.sequenceLength;
  const numFeatures = featureCols.length;
  
  let model = createModel(sequenceLength, numFeatures);
  let memory = new ReplayMemory(MEMORY_SIZE);
  let epsilon = EPSILON_START;

  console.log("\n--- Starting Training Loop ---");

  for (let e = 0; e < NUM_EPISODES; e++) {
    let state = env.reset();
    let totalReward = 0;

    while (true) {
      // --- Action Selection ---
      let action;
      if (Math.random() < epsilon) {
        // Explore: take a random action
        action = Math.floor(Math.random() * Object.keys(ACTIONS).length);
      } else {
        // Exploit: use the model to predict the best action
        action = tf.tidy(() => {
          const q_values = model.predict(state.expandDims(0));
          return q_values.argMax(1).dataSync()[0];
        });
      }
      
      // --- Interact with Environment ---
      const { state: nextState, reward, done } = env.step(action);
      totalReward += reward;

      // --- Store Experience ---
      if (nextState) {
        memory.push(state, action, reward, nextState, done);
      }
      
      state.dispose(); // Clean up old state tensor
      state = nextState;

      // --- Train the Model ---
      if (memory.memory.length > BATCH_SIZE) {
        const batch = memory.sample(BATCH_SIZE);
        const { states, targets } = await tf.tidy(() => {
            const states = tf.concat(batch.map(b => b.state.expandDims(0)));
            const nextStates = tf.concat(batch.map(b => b.nextState.expandDims(0)));
            
            const q_values_next = model.predict(nextStates);
            const q_values_curr = model.predict(states);

            const targets = q_values_curr.arraySync();
            batch.forEach((b, i) => {
                if (b.done) {
                    targets[i][b.action] = b.reward;
                } else {
                    targets[i][b.action] = b.reward + GAMMA * q_values_next.max(1).dataSync()[i];
                }
            });
            return { states, targets: tf.tensor2d(targets) };
        });
        
        await model.fit(states, targets, { verbose: 0 });
        tf.dispose([states, targets]);
      }
      
      if (done) {
        break;
      }
    }
    
    epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);
    console.log(`Episode ${e + 1}/${NUM_EPISODES} - Final Portfolio: $${env.portfolioValue.toFixed(2)} - Epsilon: ${epsilon.toFixed(4)}`);
  }

  // --- Save the Model ---
  const modelSavePath = 'file://./orium_rl_model';
  await model.save(modelSavePath);
  console.log(`\n--- TRAINING COMPLETE ---`);
  console.log(`Final model saved to: ${modelSavePath}`);
}

// --- Execute ---
runTraining();