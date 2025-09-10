// --- Imports ---
const df = require('danfojs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  console.log("--- Using Memory-Efficient Library Functions ---");

  // --- 1. Load Data ---
  console.log("\nStep 1: Loading Cleaned Data...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  const dataframe = await df.readCSV(dataPath);
  console.log("   - Data loaded successfully.");

  // --- 2. Feature Engineering (Efficiently) ---
  console.log("\nStep 2: Engineering features using optimized functions...");

  // Feature 1: Price Change. This is the correct syntax.
  const priceChange = dataframe['Close'].pctChange(1).mul(100);
  dataframe.addColumn('Price_Change', priceChange, { inplace: true });
  console.log("   - Created 'Price_Change'.");

  // Feature 2 & 3: Simple Moving Averages
  // .rolling() is the correct, optimized method.
  const sma10 = dataframe['Close'].rolling(10).mean();
  const sma30 = dataframe['Close'].rolling(30).mean();
  dataframe.addColumn('SMA_10', sma10, { inplace: true });
  dataframe.addColumn('SMA_30', sma30, { inplace: true });
  console.log("   - Created 'SMA_10' and 'SMA_30'.");

  // Fill all resulting NaN values from the calculations with 0
  dataframe.fillna({ inplace: true, values: 0 });
  console.log("   - All features created successfully.");

  // --- 3. Verification ---
  console.log("\nStep 3: Verifying Engineered Features...");
  console.log("\nDataset Head (with all new features):");
  dataframe.head(35).print();
  
  console.log("\nDataset Tail (with all new features):");
  dataframe.tail().print();

  console.log("\nNext Step: Prepare data for the AI model (Normalization and Sequencing).");

}

// --- Execute ---
runTraining();