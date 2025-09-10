// --- Imports ---
const df = require('danfojs-node');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  
  // --- Step 1: Load Data ---
  console.log("\nStep 1: Loading Cleaned Data...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
  console.log("   - Data loaded successfully.");

  // --- Step 2: Manual Feature Engineering ---
  console.log("\nStep 2: Engineering Features...");
  const closePrices = dataframe['Close'].values;
  const numPrices = closePrices.length;
  let priceChangeValues = [0];
  for (let i = 1; i < numPrices; i++) {
    const prev = closePrices[i - 1];
    const curr = closePrices[i];
    priceChangeValues.push(prev === 0 ? 0 : ((curr - prev) / prev) * 100);
  }
  dataframe.addColumn('Price_Change', priceChangeValues, { inplace: true });
  let sma10Values = [];
  for (let i = 0; i < numPrices; i++) {
    if (i < 9) { sma10Values.push(0); }
    else { let sum = 0; for (let j = 0; j < 10; j++) { sum += closePrices[i - j]; } sma10Values.push(sum / 10); }
  }
  dataframe.addColumn('SMA_10', sma10Values, { inplace: true });
  let sma30Values = [];
  for (let i = 0; i < numPrices; i++) {
    if (i < 29) { sma30Values.push(0); }
    else { let sum = 0; for (let j = 0; j < 30; j++) { sum += closePrices[i - j]; } sma30Values.push(sum / 30); }
  }
  dataframe.addColumn('SMA_30', sma30Values, { inplace: true });
  
  // REMOVED THE FAILING .fillna() LINE. It is not needed.
  console.log("   - Features created. The data is already clean.");

  // --- Step 3: Data Preparation for AI Model ---
  console.log("\nStep 3: Preparing Data for AI Model...");

  // 3a. Select Features for Training
  const featureCols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'SMA_10', 'SMA_30'];
  let featuresDf = dataframe.loc({ columns: featureCols });
  console.log("   - Selected features for training.");

  // 3b. Normalization
  const scaler = new df.MinMaxScaler();
  scaler.fit(featuresDf);
  const scaledDf = scaler.transform(featuresDf);
  console.log("   - Normalized all features to a 0-1 scale.");

  // 3c. Sequencing
  const sequenceLength = 60; // 60 minutes of data to make one prediction
  const sequences = [];
  const scaledValues = scaledDf.values; 

  for (let i = 0; i < scaledValues.length - sequenceLength; i++) {
    sequences.push(scaledValues.slice(i, i + sequenceLength));
  }
  
  const sequenceTensor = tf.tensor3d(sequences);
  console.log("   - Created sequential data for the model.");
  
  // --- Final Verification ---
  console.log("\n--- DATA PREPARATION COMPLETE ---");
  console.log("Shape of the final data tensor:", sequenceTensor.shape);
  console.log("(Number of Samples, Timesteps per Sample, Features per Timestep)");
  console.log("\nNext Step: Define the AI model architecture and start training.");
}

// --- Execute ---
runTraining();