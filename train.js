// --- Imports ---
const df = require('danfojs-node');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  
  // --- Step 1 & 2: Load Data and Engineer Features (Fast-forwarded) ---
  console.log("\nStep 1 & 2: Loading data and engineering features...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  let dataframe = await df.readCSV(dataPath);
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
  console.log("   - Features created.");

  // --- Step 3: Data Preparation for AI Model ---
  console.log("\nStep 3: Preparing Data for AI Model...");

  // 3a. Select and Normalize Features
  const featureCols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'SMA_10', 'SMA_30'];
  let featuresDf = dataframe.loc({ columns: featureCols });
  const scaler = new df.MinMaxScaler();
  scaler.fit(featuresDf);
  const scaledDf = scaler.transform(featuresDf);
  console.log("   - Features selected and normalized.");

  // 3b. Sequencing (The EFFICIENT Way)
  console.log("   - Creating sequential data using TensorFlow operations (this will be fast)...");
  const sequenceLength = 60;
  const scaledValues = scaledDf.values;

  // tf.tidy() is a memory management tool. It will automatically clean up
  // the intermediate tensors (like scaledTensor and all the individual slices)
  // once the finalTensor is created, preventing memory leaks.
  const finalTensor = tf.tidy(() => {
    // Convert the entire dataset to a 2D Tensor once.
    const scaledTensor = tf.tensor2d(scaledValues);
    
    const sequenceTensors = [];
    // Loop through, but instead of slicing JS arrays, we slice the Tensor directly.
    // This is a much faster operation.
    for (let i = 0; i <= scaledTensor.shape[0] - sequenceLength; i++) {
      // slice([start_row, start_col], [num_rows, num_cols])
      const sequence = scaledTensor.slice([i, 0], [sequenceLength, -1]); // -1 means all columns
      sequenceTensors.push(sequence);
    }
    
    // tf.stack is a highly optimized operation to combine an array of tensors
    // into a single, higher-dimension tensor.
    return tf.stack(sequenceTensors);
  });
  
  console.log("   - Sequential data created successfully.");
  
  // --- Final Verification ---
  console.log("\n--- DATA PREPARATION COMPLETE ---");
  console.log("Shape of the final data tensor:", finalTensor.shape);
  console.log("(Number of Samples, Timesteps per Sample, Features per Timestep)");

  // Clean up the final tensor from GPU memory to be a good citizen
  finalTensor.dispose();
  console.log("\n(Final tensor disposed from memory after verification)");
  console.log("\nNext Step: Define the AI model architecture and start training.");
}

// --- Execute ---
runTraining();