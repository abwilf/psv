#!/bin/bash
# Build the Verus evaluation server Singularity container

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEF_FILE="${SCRIPT_DIR}/verus_server.def"
SIF_FILE="${SCRIPT_DIR}/verus_server.sif"

# Set cache and temp directories
# Cache can be persistent, but TMPDIR must be local (/tmp) for ownership changes
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-~/.cache/singularity}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-~/.cache/apptainer}
export APPTAINER_TMPDIR=/tmp/apptainer_tmp
export TMPDIR=/tmp/tmp_build

# Create cache directories if they don't exist
mkdir -p "${APPTAINER_CACHEDIR}"
mkdir -p "${APPTAINER_TMPDIR}"
mkdir -p "${TMPDIR}"

echo "Building Verus evaluation server container..."
echo "Definition file: ${DEF_FILE}"
echo "Output file: ${SIF_FILE}"
echo "Cache dir: ${APPTAINER_CACHEDIR}"
echo ""

# Remove old container if it exists
if [ -f "${SIF_FILE}" ]; then
    echo "Removing old container..."
    rm -f "${SIF_FILE}"
fi

# Build the container (no --fakeroot due to nodev mount option on cluster)
echo "Building container (this may take several minutes)..."
singularity build "${SIF_FILE}" "${DEF_FILE}"

echo ""
echo "Build complete!"
echo "Container: ${SIF_FILE}"
