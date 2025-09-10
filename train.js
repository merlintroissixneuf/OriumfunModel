// --- Imports ---
const tf = require('@tensorflow/tfjs-node');
const df = require('danfojs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  console.log("Loading the PREPROCESSED and CLEANED dataset...");

  // --- 1. Load Clean Data ---
  // We now point to our new, smaller, and faster file.
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  
  try {
    const startTime = Date.now();
    const dataframe = await df.readCSV(dataPath);
    const endTime = Date.now();

    console.log(`\n--- SUCCESS ---`);
    console.log(`Dataset loaded in ${(endTime - startTime) / 1000} seconds.`);
    
    // --- 2. Verification Analysis ---
    console.log("\nVerifying the cleaned dataset...");
    
    console.log("\nDataset Shape (Rows, Columns):");
    console.log(dataframe.shape); // Should be [2465558, 6]

    console.log("\nColumn Names:");
    console.log(dataframe.columns);

    console.log("\nData Types:");
    dataframe.ctypes.print();
    
    console.log("\nDataset Head (First 5 Rows):");
    dataframe.head().print();

    console.log("\nDataset Tail (Last 5 Rows):");
    dataframe.tail().print();

    console.log("\nNext Step: Feature Engineering.");

  } catch (error) {
    console.error(`\n--- ERROR ---`);
    console.error(`Failed to load or process the CLEANED CSV file from: ${dataPath}`);
    console.error(error);
  }
}

// --- Execute ---
runTraining();