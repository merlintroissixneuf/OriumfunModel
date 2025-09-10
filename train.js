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
  // CORRECTED: Manually calculate percentage change.
  // .diff(1) calculates the difference between the current and previous element.
  const priceDiff = dataframe['Close'].diff(1); 
  const prevClose = dataframe['Close'].shift(1); // Gets the previous 'Close' price for each row
  const priceChange = priceDiff.div(prevClose).mul(100); // (diff / prev_close) * 100
  
  dataframe.addColumn('Price_Change', priceChange, { inplace: true });
  console.log("   - Created 'Price_Change' feature.");

  // Fill the first row's NaN value (which has no previous price) with 0
  dataframe.fillna({ columns: ['Price_Change'], values: [0], inplace: true });


  // --- 3. Verification ---
  console.log("\nStep 3: Verifying Engineered Features...");

  console.log("\nData Types (with new feature):");
  dataframe.ctypes.print();

  console.log("\nDataset Head (with Price_Change):");
  dataframe.head().print();
  
  console.log("\nDataset Tail (with Price_Change):");
  dataframe.tail().print();

  console.log("\nNext Step: Add more advanced features (Moving Averages, RSI).");

}

// --- Execute ---
runTraining();