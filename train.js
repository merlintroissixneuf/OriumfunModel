// --- Imports ---
const df = require('danfojs-node');
const tf = require('@tensorflow/tfjs-node'); // We will use TensorFlow for this part
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  
  // (Previous steps are assumed complete and correct)
  // --- Load Data ---
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
  
  // --- Manual Feature Engineering ---
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
  dataframe.fillna({ inplace: true, values: 0 });

  console.log("\n--- Data Preparation for AI Model ---");

  // --- 1. Select Features for Training ---
  // We will use these columns as input to the model. We drop the Timestamp.
  const featureCols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'SMA_10', 'SMA_30'];
  let featuresDf = dataframe.loc({ columns: featureCols });
  console.log("Step 1: Selected features for training.");

  // --- 2. Normalization ---
  // Scale all feature values to be between 0 and 1.
  const scaler = new df.MinMaxScaler();
  scaler.fit(featuresDf);
  const scaledDf = scaler.transform(featuresDf);
  console.log("Step 2: Normalized all features to a 0-1 scale.");

  console.log("\nVerification of Scaled Data (First 5 rows):");
  scaledDf.head().print();

  // --- 3. Sequencing ---
  // Convert the dataframe into an array of sequences.
  const sequenceLength = 60; // We will use 60 minutes of data to predict the future.
  const sequences = [];
  const scaledValues = scaledDf.values; // Get the data as a 2D array

  for (let i = 0; i < scaledValues.length - sequenceLength; i++) {
    sequences.push(scaledValues.slice(i, i + sequenceLength));
  }
  
  // Convert the array of sequences into a TensorFlow Tensor
  const sequenceTensor = tf.tensor3d(sequences);

  console.log("\nStep 3: Created sequential data for the model.");
  console.log("   - Sequence length: 60 minutes");
  
  console.log("\n--- Verification of Final Tensor ---");
  console.log("Shape of the final data tensor:", sequenceTensor.shape);
  console.log("(Number of Samples, Timesteps per Sample, Features per Timestep)");

  console.log("\n--- DATA PREPARATION COMPLETE ---");
  console.log("Next Step: Define the AI model architecture and start training.");
}

// --- Execute ---
runTraining();