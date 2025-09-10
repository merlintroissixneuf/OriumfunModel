// --- Imports ---
const fs = require('fs');
const readline = require('readline');
const path = require('path');

// --- Configuration ---
const INPUT_FILE = 'data/BTCUSD_1min.csv';
const OUTPUT_FILE = 'data/BTCUSD_1min_clean.csv';
// It's MUCH faster to compare numbers than to create Date objects for every line.
// This is the UNIX timestamp for January 1st, 2021 00:00:00 UTC.
const START_TIMESTAMP = 1609459200; 

async function runStreamPreprocessing() {
  console.log("--- Starting Orium Data Stream Preprocessing ---");
  console.log("This method reads the file line-by-line to handle any file size.");
  console.log(`It will be fast and memory-efficient.\n`);

  const inputPath = path.join(__dirname, INPUT_FILE);
  const outputPath = path.join(__dirname, OUTPUT_FILE);

  try {
    // Create a read stream from the massive source file
    const fileStream = fs.createReadStream(inputPath);

    // Create a readline interface to process the stream line by line
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity // Handles different line endings automatically
    });

    // Create a write stream to our new, clean file
    const outputStream = fs.createWriteStream(outputPath);

    let isFirstLine = true;
    let processedLines = 0;
    let keptLines = 0;

    console.log("Processing stream... this may take a minute or two.");

    // This event is triggered for each line in the file
    rl.on('line', (line) => {
      processedLines++;

      if (isFirstLine) {
        // The first line is the header, we always keep it.
        outputStream.write(line + '\n');
        isFirstLine = false;
        return; // Move to the next line
      }

      // Extract the timestamp (the first value before the comma)
      const timestampStr = line.substring(0, line.indexOf(','));
      const timestamp = parseFloat(timestampStr);

      // Check if the timestamp is recent enough
      if (timestamp >= START_TIMESTAMP) {
        outputStream.write(line + '\n');
        keptLines++;
      }

      // Optional: Log progress every 500,000 lines
      if (processedLines % 500000 === 0) {
        console.log(`...scanned ${processedLines.toLocaleString()} lines...`);
      }
    });

    // This event is triggered when the whole file has been read
    rl.on('close', () => {
      console.log("\n-------------------------------------------");
      console.log("---           SUCCESS!            ---");
      console.log("---  Stream Processing Complete.  ---");
      console.log("-------------------------------------------");
      console.log(`Total lines scanned: ${processedLines.toLocaleString()}`);
      console.log(`Lines kept (since 2021): ${keptLines.toLocaleString()}`);
      console.log(`Cleaned data is ready in: ${OUTPUT_FILE}`);
    });

  } catch (error) {
    console.error(`\n\n--- FATAL ERROR ---`);
    console.error(`An error occurred during stream processing.`);
    console.error(error);
  }
}

// --- Execute ---
runStreamPreprocessing();