#!/bin/bash

# X-Ray Insight Lab Quick Start Script
# This script starts the application without checking or installing dependencies

echo "Quick Starting X-Ray Insight Lab..."
echo "---------------------------------"

# Determine Python command
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Start the backend server
echo "Starting backend server..."
cd server

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

$PYTHON_CMD run.py &
BACKEND_PID=$!
echo "✅ Backend server started (PID: $BACKEND_PID)"

# Go back to the main directory
cd ..

# Start the frontend
echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "X-Ray Insight Lab is now running!"
echo "--------------------------------"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173 (or check the URL in the terminal output)"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped"
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup INT

# Wait for user to press Ctrl+C
wait
