// --- Imports ---
const df = require('danfojs-node');
const path = require('path');

// --- Configuration ---
const INPUT_FILE = 'data/BTCUSD_1min.csv';
const OUTPUT_FILE = 'data/BTCUSD_1min_clean.csv';
const START_DATE = '2021-01-01'; // We'll only use data from this date forward

async function runPreprocessing() {
  console.log("Starting preprocessing of the massive dataset...");
  console.log(`This may take a few minutes, but it's a one-time operation.\n`);

  const inputPath = path.join(__dirname, INPUT_FILE);
  const outputPath = path.join(__dirname, OUTPUT_FILE);

  try {
    // --- 1. Load the massive CSV ---
    console.log(`Step 1: Loading the huge CSV file: ${INPUT_FILE}...`);
    let dataframe = await df.readCSV(inputPath);
    console.log(`Successfully loaded ${dataframe.shape[0]} rows.`);

    // --- 2. Data Cleaning & Selection ---
    console.log("\nStep 2: Cleaning and selecting relevant data...");
    
    // Convert 'Unix' or 'Date' column to a proper datetime object for filtering
    // IMPORTANT: Check your CSV's header. It might be 'Unix', 'Date', 'unix', etc.
    // Let's assume the column is named 'Date' for now.
    const dateColumnName = 'Date'; // <--- YOU MIGHT NEED TO CHANGE THIS
    dataframe = dataframe.asType(dateColumnName, 'datetime');
    
    // Filter the dataframe to only include data after our START_DATE
    const recentDataframe = dataframe.query({
      column: dateColumnName,
      is: '>=',
      to: START_DATE,
    });
    console.log(`Filtered down to ${recentDataframe.shape[0]} rows from ${START_DATE} onwards.`);

    // Select only the columns we actually need for training
    const finalDataframe = recentDataframe.loc({
      columns: [dateColumnName, 'Open', 'High', 'Low', 'Close', 'Volume BTC'], // <--- CHECK 'Volume BTC' or similar
    });

    console.log("Final selected columns:", finalDataframe.columns);

    // --- 3. Save the Cleaned Data ---
    console.log(`\nStep 3: Saving the cleaned data to: ${OUTPUT_FILE}...`);
    await df.toCSV(finalDataframe, { filePath: outputPath });
    console.log("\n--- SUCCESS ---");
    console.log(`Preprocessing complete! You can now use the smaller, faster file for training.`);

  } catch (error) {
    console.error(`\n--- ERROR ---`);
    console.error(`An error occurred during preprocessing.`);
    console.error("Check that the column names in the script (e.g., 'Date', 'Volume BTC') exactly match your CSV file's header.");
    console.error(error);
  }
}

// --- Execute ---
runPreprocessing();