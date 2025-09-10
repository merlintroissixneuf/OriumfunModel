// --- Imports ---
const df = require('danfojs-node');
// THIS IS THE CORRECTED LINE. The package is '@tensorflow/tfjs-node'.
const tf = require('@tensorflow/tfjs-node'); 
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  console.log("--- FINAL: Using the tf.data.Dataset API for maximum performance. ---");
  
  // --- Step 1 & 2: Load Data and Engineer Features (Combined for brevity) ---
  console.log("\nStep 1 & 2: Loading data and engineering features...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
  const closePrices = dataframe['Close'].values;
  const numPrices = closePrices.length;
  let priceChangeValues = [0]; for (let i = 1; i < numPrices; i++) { const p = closePrices[i - 1], c = closePrices[i]; priceChangeValues.push(p === 0 ? 0 : ((c - p) / p) * 100); } dataframe.addColumn('Price_Change', priceChangeValues, { inplace: true });
  let sma10Values = []; for (let i = 0; i < numPrices; i++) { if (i < 9) { sma10Values.push(0); } else { let s = 0; for (let j = 0; j < 10; j++) { s += closePrices[i - j]; } sma10Values.push(s / 10); } } dataframe.addColumn('SMA_10', sma10Values, { inplace: true });
  let sma30Values = []; for (let i = 0; i < numPrices; i++) { if (i < 29) { sma30Values.push(0); } else { let s = 0; for (let j = 0; j < 30; j++) { s += closePrices[i - j]; } sma30Values.push(s / 30); } } dataframe.addColumn('SMA_30', sma30Values, { inplace: true });
  console.log("   - Features created.");

  // --- Step 3: Data Preparation using the tf.data API ---
  console.log("\nStep 3: Preparing Data with the TensorFlow Data Pipeline...");

  // 3a. Select and Normalize Features
  const featureCols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'SMA_10', 'SMA_30'];
  let featuresDf = dataframe.loc({ columns: featureCols });
  const scaler = new df.MinMaxScaler();
  scaler.fit(featuresDf);
  const scaledDf = scaler.transform(featuresDf);
  console.log("   - Features selected and normalized.");

  // 3b. Sequencing (The Professional Way)
  console.log("   - Creating sequential data using the tf.data API (this will be fast)...");
  const sequenceLength = 60;
  
  const finalTensor = await tf.tidy(() => {
    const scaledTensor = tf.tensor2d(scaledDf.values);
    const dataset = tf.data.Dataset.fromTensorSlices(scaledTensor);
    const windowedDataset = dataset.window(sequenceLength, 1, 1, true);
    const flatDataset = windowedDataset.flatMap(window => window.batch(sequenceLength));
    return flatDataset.toArray();
  }).then(tensorArray => {
    return tf.stack(tensorArray);
  });
  
  console.log("   - Sequential data created successfully.");
  
  // --- Final Verification ---
  console.log("\n--- DATA PREPARATION COMPLETE ---");
  console.log("Shape of the final data tensor:", finalTensor.shape);
  console.log("(Number of Samples, Timesteps per Sample, Features per Timestep)");

  finalTensor.dispose();
  console.log("\n(Final tensor disposed from memory after verification)");
  console.log("\nTHE DATA IS READY. Next Step: Defining and Training the Model.");
}

// --- Execute ---
runTraining();