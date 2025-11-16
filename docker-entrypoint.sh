#!/bin/bash
# Entrypoint script that ensures video recording cleanup on shutdown

set -e

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Container stopping - finalizing video recording..."
    
    # Send SIGTERM to the Python process to trigger cleanup
    if [ ! -z "$PYTHON_PID" ]; then
        kill -TERM "$PYTHON_PID" 2>/dev/null || true
        # Wait for Python process to finish (max 10 seconds)
        for i in {1..10}; do
            if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
    fi
    
    echo "✅ Cleanup complete"
    exit 0
}

# Register cleanup function to run on SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

# Run the command passed to this script and capture its PID
"$@" &
PYTHON_PID=$!

# Wait for the Python process
wait $PYTHON_PID
EXIT_CODE=$?

exit $EXIT_CODE
