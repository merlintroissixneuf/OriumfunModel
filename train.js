// --- Imports ---
const df = require('danfojs-node');
const path = require('path');

// --- Main Function ---
async function runTraining() {
  console.log("--- Orium v2.0 Training Process ---");
  console.log("--- CRITICAL: Using 100% Manual Calculation Mode for Guaranteed Execution ---");

  // --- 1. Load Data ---
  console.log("\nStep 1: Loading Cleaned Data...");
  const dataPath = path.join(__dirname, 'data/BTCUSD_1min_clean.csv');
  const dataframe = await df.readCSV(dataPath);
  console.log("   - Data loaded successfully.");

  // --- 2. Manual Feature Engineering ---
  console.log("\nStep 2: Engineering Features with Manual Loops...");

  // Get the 'Close' prices as a standard JavaScript array to prevent library issues.
  const closePrices = dataframe['Close'].values;
  const numPrices = closePrices.length;

  // --- Manual Calculation for Price_Change ---
  console.log("   - Manually calculating 'Price_Change'...");
  let priceChangeValues = [0]; // Start with 0 for the first element.
  for (let i = 1; i < numPrices; i++) {
    const prev = closePrices[i - 1];
    const curr = closePrices[i];
    // Handle potential division by zero, though unlikely with price data.
    const change = prev === 0 ? 0 : ((curr - prev) / prev) * 100;
    priceChangeValues.push(change);
  }
  dataframe.addColumn('Price_Change', priceChangeValues, { inplace: true });

  // --- Manual Calculation for SMA_10 ---
  console.log("   - Manually calculating 'SMA_10'...");
  let sma10Values = [];
  for (let i = 0; i < numPrices; i++) {
    if (i < 9) {
      sma10Values.push(0); // Not enough data for the first 9 elements.
    } else {
      let sum = 0;
      for (let j = 0; j < 10; j++) {
        sum += closePrices[i - j];
      }
      sma10Values.push(sum / 10);
    }
  }
  dataframe.addColumn('SMA_10', sma10Values, { inplace: true });

  // --- Manual Calculation for SMA_30 ---
  console.log("   - Manually calculating 'SMA_30'...");
  let sma30Values = [];
  for (let i = 0; i < numPrices; i++) {
    if (i < 29) {
      sma30Values.push(0); // Not enough data for the first 29 elements.
    } else {
      let sum = 0;
      for (let j = 0; j < 30; j++) {
        sum += closePrices[i - j];
      }
      sma30Values.push(sum / 30);
    }
  }
  dataframe.addColumn('SMA_30', sma30Values, { inplace: true });
  
  console.log("   - All features created successfully.");

  // --- 3. Verification ---
  console.log("\nStep 3: Verifying Engineered Features...");
  console.log("\nDataset Head (with all new features):");
  dataframe.head(35).print();
  
  console.log("\nDataset Tail (with all new features):");
  dataframe.tail().print();

  console.log("\nThis Phase is Complete. Next Step: Normalization and Sequencing.");
}

// --- Execute ---
runTraining();