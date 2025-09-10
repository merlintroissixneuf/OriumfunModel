// --- Imports ---
const df = require('danfojs-node');
const path = require('path');

// --- Configuration ---
const INPUT_FILE = 'data/BTCUSD_1min.csv';
const OUTPUT_FILE = 'data/BTCUSD_1min_clean.csv';
const START_DATE = '2021-01-01';

// --- Helper for showing progress ---
const spinner = {
  chars: ['|', '/', '-', '\\'],
  timer: null,
  index: 0,
  start: function(message) {
    process.stdout.write(message + ' ');
    this.timer = setInterval(() => {
      process.stdout.write(this.chars[this.index++]);
      this.index %= this.chars.length;
      process.stdout.cursorTo(process.stdout.cursorTo.length - 1);
    }, 250);
  },
  stop: function() {
    clearInterval(this.timer);
    process.stdout.write(' DONE\n');
  }
};

async function runPreprocessing() {
  console.log("--- Starting Orium Data Preprocessing ---");
  console.log(`This is a one-time operation that will run in the background.\n`);

  const inputPath = path.join(__dirname, INPUT_FILE);
  const outputPath = path.join(__dirname, OUTPUT_FILE);
  let dataframe;

  try {
    // --- 1. Load the massive CSV ---
    spinner.start(`Step 1: Loading huge CSV file (${INPUT_FILE}). This is the longest step...`);
    dataframe = await df.readCSV(inputPath);
    spinner.stop();
    console.log(`   - Successfully loaded ${dataframe.shape[0]} total rows.`);

    // --- 2. Data Cleaning & Selection ---
    console.log(`\nStep 2: Cleaning and selecting data from ${START_DATE} onwards...`);
    
    const dateColumn = dataframe['Timestamp'].apply((ts) => new Date(ts * 1000));
    dataframe.addColumn('Date', dateColumn, { inplace: true });

    const recentDataframe = dataframe.query({
      column: 'Date',
      is: '>=',
      to: START_DATE,
    });
    console.log(`   - Filtered down to ${recentDataframe.shape[0]} relevant rows.`);

    const finalDataframe = recentDataframe.loc({
      columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
    });
    console.log(`   - Selected final columns: ${finalDataframe.columns}`);
    
    // --- 3. Save the Cleaned Data ---
    spinner.start(`\nStep 3: Saving cleaned data to ${OUTPUT_FILE}...`);
    await df.toCSV(finalDataframe, { filePath: outputPath });
    spinner.stop();

    console.log("\n-------------------------------------------");
    console.log("---           SUCCESS!            ---");
    console.log("--- Preprocessing is complete.    ---");
    console.log("-------------------------------------------");

  } catch (error) {
    spinner.stop(); // Ensure spinner stops on error
    console.error(`\n\n--- FATAL ERROR ---`);
    console.error(`An error occurred during preprocessing.`);
    console.error(error);
  }
}

// --- Execute ---
runPreprocessing();