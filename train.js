// --- Imports ---
const df = require('danfojs-node');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { TradingEnvironment, ACTIONS } = require('./trading_env.js');
const cliProgress = require('cli-progress');

// --- Hyperparameters ---
const GAMMA = 0.95, EPSILON_START = 1.0, EPSILON_END = 0.01, EPSILON_DECAY = 0.995;
const LEARNING_RATE = 0.001, MEMORY_SIZE = 10000, BATCH_SIZE = 64, NUM_EPISODES = 50;

// --- The Agent's "Brain" ---
function createModel(sequenceLength, numFeatures) {
  const model = tf.sequential();
  model.add(tf.layers.reshape({ inputShape: [sequenceLength, numFeatures], targetShape: [sequenceLength, numFeatures, 1] }));
  model.add(tf.layers.conv1d({ kernelSize: 5, filters: 32, activation: 'relu' }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(tf.layers.lstm({ units: 50 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 3, activation: 'linear' })); 
  model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
  return model;
}

// --- The Agent's "Memory" ---
class ReplayMemory {
  constructor(capacity) { this.capacity = capacity; this.memory = []; this.position = 0; }
  push(state, action, reward, nextState, done) { if (this.memory.length < this.capacity) { this.memory.push(null); } this.memory[this.position] = { state, action, reward, nextState, done }; this.position = (this.position + 1) % this.capacity; }
  sample(batchSize) { const samples = []; for (let i = 0; i < batchSize; i++) { const index = Math.floor(Math.random() * this.memory.length); samples.push(this.memory[index]); } return samples; }
}

// --- Main Training Function ---
async function runTraining() {
  console.log("--- Orium Reinforcement Learning Engine ---");

  // Step 1 & 2: Load Data and Engineer Features
  console.log("\nStep 1 & 2: Loading data and engineering features...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
  const closePrices = dataframe['Close'].values;
  const numPrices = closePrices.length;
  const progressBar = new cliProgress.SingleBar({ format: ' {bar} | {percentage}% | ETA: {eta}s | {value}/{total} | {task}' }, cliProgress.Presets.shades_classic);
  let priceChangeValues = [0]; progressBar.start(numPrices, 0, { task: "Calculating Price_Change" }); for (let i = 1; i < numPrices; i++) { const p=closePrices[i-1], c=closePrices[i]; priceChangeValues.push(p===0?0:((c-p)/p)*100); progressBar.increment(); } progressBar.stop(); dataframe.addColumn('Price_Change', priceChangeValues, { inplace: true });
  let sma10Values = []; progressBar.start(numPrices, 0, { task: "Calculating SMA_10      " }); for (let i=0; i<numPrices; i++) { if (i<9) { sma10Values.push(0); } else { let s=0; for (let j=0; j<10; j++) { s+=closePrices[i-j]; } sma10Values.push(s/10); } progressBar.increment(); } progressBar.stop(); dataframe.addColumn('SMA_10', sma10Values, { inplace: true });
  let sma30Values = []; progressBar.start(numPrices, 0, { task: "Calculating SMA_30      " }); for (let i=0; i<numPrices; i++) { if (i<29) { sma30Values.push(0); } else { let s=0; for (let j=0; j<30; j++) { s+=closePrices[i-j]; } sma30Values.push(s/30); } progressBar.increment(); } progressBar.stop(); dataframe.addColumn('SMA_30', sma30Values, { inplace: true });
  dataframe.dropNa({ inplace: true });

  // Step 3: Manual Normalization with Progress Bars
  console.log("\nStep 3: Normalizing data for the AI...");
  const featureCols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'SMA_10', 'SMA_30'];
  const featuresDf = dataframe.loc({ columns: featureCols });
  const minMax = {}; featureCols.forEach(col => { minMax[col] = { min: Infinity, max: -Infinity } });
  progressBar.start(featuresDf.shape[0], 0, { task: "Finding Min/Max Values  " });
  for (let i = 0; i < featuresDf.shape[0]; i++) {
    for (const col of featureCols) { const val = featuresDf.iat(i, featureCols.indexOf(col)); if (val < minMax[col].min) minMax[col].min = val; if (val > minMax[col].max) minMax[col].max = val; }
    progressBar.increment();
  }
  progressBar.stop();
  const scaledData = {}; featureCols.forEach(col => scaledData[col] = []);
  progressBar.start(featuresDf.shape[0], 0, { task: "Applying Normalization " });
  for (let i = 0; i < featuresDf.shape[0]; i++) {
    for (const col of featureCols) { const val = featuresDf.iat(i, featureCols.indexOf(col)); const min=minMax[col].min, max=minMax[col].max; const range=max-min; scaledData[col].push(range===0?0:(val-min)/range); }
    progressBar.increment();
  }
  progressBar.stop();
  let scaledDf = new df.DataFrame(scaledData);
  scaledDf.addColumn('Close', dataframe['Close'], { inplace: true });

  // Step 4: Initialize Environment and Agent
  console.log("\nStep 4: Initializing Trading Environment...");
  const env = new TradingEnvironment(scaledDf);
  const sequenceLength = env.sequenceLength; const numFeatures = featureCols.length;
  let model = createModel(sequenceLength, numFeatures);
  let memory = new ReplayMemory(MEMORY_SIZE);
  let epsilon = EPSILON_START;

  console.log("\n--- Starting Training Loop ---");
  for (let e = 0; e < NUM_EPISODES; e++) {
    let state = env.reset(); let totalReward = 0; let step = 0;
    while (true) {
      step++; let action;
      if (Math.random() < epsilon) { action = Math.floor(Math.random() * Object.keys(ACTIONS).length); }
      else { action = tf.tidy(() => { const q=model.predict(state.expandDims(0)); return q.argMax(1).dataSync()[0]; }); }
      const { state: nextState, reward, done } = env.step(action);
      totalReward += reward;
      if (nextState) { memory.push(state, action, reward, nextState, done); }
      state.dispose(); state = nextState;
      if (memory.memory.length > BATCH_SIZE) {
        const batch = memory.sample(BATCH_SIZE);
        const { states, targets } = await tf.tidy(() => {
            const states=tf.concat(batch.map(b=>b.state.expandDims(0))); const nextStates=tf.concat(batch.map(b=>b.nextState.expandDims(0)));
            const q_values_next=model.predict(nextStates); const q_values_curr=model.predict(states);
            const targets = q_values_curr.arraySync();
            batch.forEach((b, i) => { if (b.done) { targets[i][b.action] = b.reward; } else { targets[i][b.action] = b.reward + GAMMA * q_values_next.max(1).dataSync()[i]; } });
            return { states, targets: tf.tensor2d(targets) };
        });
        await model.fit(states, targets, { verbose: 0 }); tf.dispose([states, targets]);
      }
      if (done) { break; }
    }
    epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);
    console.log(`Episode ${e + 1}/${NUM_EPISODES} - Final Portfolio: $${env.portfolioValue.toFixed(2)} - Steps: ${step} - Epsilon: ${epsilon.toFixed(4)}`);
  }
  const modelSavePath = 'file://./orium_rl_model'; await model.save(modelSavePath);
  console.log(`\n--- TRAINING COMPLETE ---`); console.log(`Final model saved to: ${modelSavePath}`);
}
runTraining();