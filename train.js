// This is a test file to confirm we can use our AI libraries in the Cloud Shell.

console.log("Attempting to load AI libraries...");

try {
  const tf = require('@tensorflow/tfjs-node');
  const df = require('danfojs-node');

  console.log("Success! TensorFlow and Danfo.js are ready.");
  
  // We can even print the versions to be sure
  console.log(`TensorFlow.js version: ${tf.version.tfjs}`);
  
} catch (error) {
  console.error("Failed to load AI libraries. Make sure you have run 'npm install @tensorflow/tfjs-node danfojs-node' in this environment.");
  console.error(error);
}  