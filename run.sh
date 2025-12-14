#!/bin/bash
# Port-to-Rail Surge Forecaster - Quick Start Script

echo "🚢 Port-to-Rail Surge Forecaster"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the logistics directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Check for data
if [ ! -d "data" ]; then
    echo "⚠️ Warning: 'data' directory not found"
    echo "   Make sure to place data files in the 'data' folder"
fi

# Check for output data (pre-trained)
if [ -f "output/port_terminal_mapping.csv" ]; then
    echo "✅ Pre-processed data found"
else
    echo "⚠️ No pre-processed data found"
    echo "   Run 'python train_champion_model.py' to generate"
fi

echo ""
echo "🚀 Starting API server..."
echo "   Dashboard: Open frontend/index.html in browser"
echo "   API Docs:  http://localhost:8000/docs"
echo ""

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000
