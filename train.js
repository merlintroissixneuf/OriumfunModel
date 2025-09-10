// --- Imports ---
const df = require('danfojs-node');
const tf = require('@tensorflow/js-node'); // Correct import for tfjs-node
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  console.log("--- FINAL ATTEMPT: Using the tf.data.Dataset API for maximum performance. ---");
  
  // --- Step 1 & 2: Load Data and Engineer Features (Combined for brevity) ---
  console.log("\nStep 1 & 2: Loading data and engineering features...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
  const closePrices = dataframe['Close'].values;
  const numPrices = closePrices.length;
  // (Manual feature engineering loops are confirmed to be fast enough)
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
  console.log("   - Creating sequential data using the tf.data API...");
  const sequenceLength = 60;
  
  // This entire operation is now handed off to the highly-optimized TensorFlow engine.
  // There are NO JavaScript loops involved in the windowing process.
  const finalTensor = await tf.tidy(() => {
    // 1. Convert our data to a single 2D Tensor
    const scaledTensor = tf.tensor2d(scaledDf.values);
    
    // 2. Create a dataset where each element is one row of the tensor
    const dataset = tf.data.Dataset.fromTensorSlices(scaledTensor);
    
    // 3. Create sliding windows. This is the core of the operation.
    // It creates a dataset of datasets, where each sub-dataset is a window.
    const windowedDataset = dataset.window(sequenceLength, 1, 1, true);
    
    // 4. Flatten the dataset of datasets into a dataset of tensors.
    // Each element of the stream is now a 2D tensor of shape [60, num_features]
    const flatDataset = windowedDataset.flatMap(window => window.batch(sequenceLength));
    
    // 5. Collect all the 2D tensors from the stream into a single JavaScript array
    return flatDataset.toArray();
  }).then(tensorArray => {
    // 6. Stack the array of 2D tensors into our final 3D tensor.
    // This is done after the tidy block to return the final result.
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