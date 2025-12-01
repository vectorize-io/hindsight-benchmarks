#!/bin/bash
# Start the benchmark visualizer from the project root

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📊 Hindsight Benchmarks Visualizer"
echo "=================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if results directory exists
if [ ! -d "$PROJECT_ROOT/results" ]; then
    echo "⚠️  Warning: results/ directory not found"
    echo "   Please run benchmarks first to generate results"
    echo ""
fi

# Check if longmemeval results exist
if [ -f "$PROJECT_ROOT/results/longmemeval.json" ]; then
    echo "✓ Found LongMemEval results"
else
    echo "⚠️  LongMemEval results not found (results/longmemeval.json)"
fi

# Check if locomo results exist
if [ -f "$PROJECT_ROOT/results/locomo.json" ]; then
    echo "✓ Found LoComo results"
else
    echo "ℹ️  LoComo results not found (will be added later)"
fi

echo ""
echo "Starting visualizer..."
echo ""

# Change to visualizer directory and run via UV
cd "$PROJECT_ROOT/visualizer"

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "📦 Installing dependencies..."
    uv sync
fi

# Run the visualizer using the UV script
exec uv run visualizer
