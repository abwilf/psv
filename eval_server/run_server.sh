#!/bin/bash
# Launch the Verus evaluation server in a Singularity container

# FAIL LOUDLY if PHYSICAL_GPU not set
if [ -z "${PHYSICAL_GPU}" ] || [ "${PHYSICAL_GPU}" = "ERROR_NOT_SET" ]; then
    echo "❌ ERROR: PHYSICAL_GPU environment variable is not set!"
    echo ""
    echo "You must set PHYSICAL_GPU before running this script."
    echo "To detect your physical GPU index, run:"
    echo "  python3 -c \"from src.utils.gpu_utils import get_physical_gpu_index; print(get_physical_gpu_index())\""
    echo ""
    echo "Then set it in your script:"
    echo "  export PHYSICAL_GPU=4  # (use the detected index)"
    echo ""
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONTAINER_PATH="${SCRIPT_DIR}/verus_server.sif"

if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "Error: Container not found at ${CONTAINER_PATH}"
    echo "Please build the container first with:"
    echo "  singularity build ${CONTAINER_PATH} ${SCRIPT_DIR}/verus_server.def"
    exit 1
fi

# Use port from environment (calculated in experiments.sh)
PORT="${EVAL_SERVER_PORT:-5000}"

# Use GPU-specific scratch directory to avoid conflicts
SCRATCH_DIR="${SCRIPT_DIR}/scratch/gpu_${PHYSICAL_GPU}"

# Create scratch directory on host if it doesn't exist
mkdir -p "${SCRATCH_DIR}"

echo "Starting Verus evaluation server..."
echo "Container: ${CONTAINER_PATH}"
echo "Server directory: ${SCRIPT_DIR}"
echo "Physical GPU: ${PHYSICAL_GPU}"
echo "Port: ${PORT}"
echo "Scratch directory: ${SCRATCH_DIR}"
echo ""
echo "Server will be available at http://localhost:${PORT}"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the container with both the server directory and verus bound
singularity run \
    --bind "${SCRIPT_DIR}:/server" \
    --bind "${VERUS_PATH:-/opt/verus}:/verus" \
    --bind "${SCRATCH_DIR}:/scratch" \
    --env "EVAL_SERVER_PORT=${PORT}" \
    --cleanenv \
    "${CONTAINER_PATH}"
