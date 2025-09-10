// --- Imports ---
const tf = require('@tensorflow/tfjs-node');
const df = require('danfojs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");

  // --- 1. Load Data ---
  console.log("\nStep 1: Loading Cleaned Data...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  const dataframe = await df.readCSV(dataPath);
  console.log("   - Data loaded successfully.");

  // --- 2. Feature Engineering ---
  console.log("\nStep 2: Engineering New Features...");

  // Feature 1: Price Change
  console.log("   - Calculating 'Price_Change'...");
  // THE CORRECT WAY: .shift() is a method of the DataFrame, not the Series.
  const prev_df = dataframe.shift(1); // Create a new DataFrame with all data shifted down
  const prevClose = prev_df['Close'];     // Select the shifted 'Close' column from the new DataFrame
  const priceChange = dataframe['Close'].sub(prevClose).div(prevClose).mul(100);
  dataframe.addColumn('Price_Change', priceChange, { inplace: true });
  
  // Feature 2 & 3: Simple Moving Averages (SMA)
  console.log("   - Calculating 'SMA_10' and 'SMA_30'...");
  // .rolling() is the correct method on a Series for moving-window calculations.
  const sma10 = dataframe['Close'].rolling(10).mean();
  const sma30 = dataframe['Close'].rolling(30).mean();
  dataframe.addColumn('SMA_10', sma10, { inplace: true });
  dataframe.addColumn('SMA_30', sma30, { inplace: true });

  // Fill all resulting NaN values from the calculations with 0
  dataframe.fillna({ inplace: true, values: 0 });
  console.log("   - All initial features created.");

  // --- 3. Verification ---
  console.log("\nStep 3: Verifying Engineered Features...");
  console.log("\nDataset Head (with all new features):");
  dataframe.head(15).print(); // Print more rows to see the SMAs start to populate
  
  console.log("\nDataset Tail (with all new features):");
  dataframe.tail().print();

  console.log("\nNext Step: Prepare the data for the AI model (Normalization and Sequencing).");

}

// --- Execute ---
runTraining();