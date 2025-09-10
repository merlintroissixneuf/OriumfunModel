// --- Imports ---
const tf = require('@tensorflow/tfjs-node');
const df = require('danfojs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("Starting Orium v2.0 training process...");

  // --- 1. Load Data ---
  console.log("Step 1: Loading historical data...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min.csv');
  
  try {
    const dataframe = await df.readCSV(dataPath);
    console.log("Successfully loaded the dataset.");

    // --- 2. Initial Analysis ---
    console.log("\nStep 2: Performing initial analysis...");
    
    // Print the first 5 rows to see what the data looks like
    console.log("Dataset Head (First 5 Rows):");
    dataframe.head().print();

    // Print the shape of the data (rows, columns)
    console.log("\nDataset Shape:");
    console.log(dataframe.shape);

    // Print the column names
    console.log("\nColumn Names:");
    console.log(dataframe.columns);

    // Print data types of each column
    console.log("\nData Types:");
    dataframe.ctypes.print();


  } catch (error) {
    console.error(`\n--- ERROR ---`);
    console.error(`Failed to load or process the CSV file from: ${dataPath}`);
    console.error("Please ensure the file exists and is correctly named 'BTCUSD_1min.csv' inside the 'data' directory.");
    console.error(error);
  }
}

// --- Execute ---
runTraining();