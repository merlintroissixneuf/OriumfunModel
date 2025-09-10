// --- Imports ---
const express = require('express');

// --- Initialization ---
// Create an instance of the Express application
const app = express();

// Define the port. Render will provide the PORT variable.
// For local development, we'll default to 3001.
const PORT = process.env.PORT || 3001;

// --- Middleware ---
// This allows our server to understand incoming JSON data
app.use(express.json());

// --- Routes ---
// This is the main "health check" route.
// When you visit the base URL of our API, it will send back a success message.
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    project: 'Orium Backend v1.0'
  });
});

// --- Server Startup ---
// This starts the server and makes it listen for requests on the defined port.
app.listen(PORT, () => {
  console.log(`Orium server is running on port ${PORT}`);
});