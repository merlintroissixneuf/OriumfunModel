// --- Imports ---
const df = require('danfojs-node');
const path = require('path');

// --- Configuration ---
const INPUT_FILE = 'data/BTCUSD_1min.csv';
const OUTPUT_FILE = 'data/BTCUSD_1min_clean.csv';
const START_DATE = '2021-01-01'; // We only care about data from this date forward.

async function runPreprocessing() {
  console.log("Starting preprocessing of the massive dataset...");
  console.log(`This is a one-time operation and may take a few minutes.\n`);

  const inputPath = path.join(__dirname, INPUT_FILE);
  const outputPath = path.join(__dirname, OUTPUT_FILE);

  try {
    // --- 1. Load the massive CSV ---
    console.log(`Step 1: Loading the huge CSV file: ${INPUT_FILE}...`);
    let dataframe = await df.readCSV(inputPath);
    console.log(`Successfully loaded ${dataframe.shape[0]} rows.`);

    // --- 2. Data Cleaning & Selection ---
    console.log("\nStep 2: Cleaning and selecting relevant data...");
    
    // CORRECTED: Create a proper JavaScript Date object from the UNIX timestamp.
    // UNIX timestamps are in seconds, JavaScript Date needs milliseconds, so we multiply by 1000.
    const dateColumn = dataframe['Timestamp'].apply((ts) => new Date(ts * 1000));
    dataframe.addColumn('Date', dateColumn, { inplace: true });

    // Filter the dataframe to only include data after our START_DATE
    const recentDataframe = dataframe.query({
      column: 'Date',
      is: '>=',
      to: START_DATE,
    });
    console.log(`Filtered down to ${recentDataframe.shape[0]} rows from ${START_DATE} onwards.`);

    // CORRECTED: Select only the columns we actually need, using the exact names from your file.
    const finalDataframe = recentDataframe.loc({
      columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
    });

    console.log("Final selected columns:", finalDataframe.columns);
    console.log("Data Head (First 5 Rows of Cleaned Data):");
    finalDataframe.head().print();

    // --- 3. Save the Cleaned Data ---
    console.log(`\nStep 3: Saving the cleaned data to: ${OUTPUT_FILE}...`);
    await df.toCSV(finalDataframe, { filePath: outputPath });
    console.log("\n--- SUCCESS ---");
    console.log(`Preprocessing complete! The new file '${OUTPUT_FILE}' is ready for training.`);

  } catch (error) {
    console.error(`\n--- ERROR ---`);
    console.error(`An error occurred during preprocessing.`);
    console.error(error);
  }
}

// --- Execute ---
runPreprocessing();