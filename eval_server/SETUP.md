# Verus Evaluation Server - Setup Guide

## Quick Start

```bash
cd eval_server

# 1. Build the container (one-time, ~5 minutes)
singularity build verus_server.sif verus_server.def

# 2. Set required environment variables
export PHYSICAL_GPU=0  # Your GPU index
export VERUS_PATH=/path/to/verus/verus-x86-linux
export EVAL_SERVER_PORT=5000

# 3. Start the server
./run_server.sh
```

## Architecture

The evaluation server runs Verus inside a Singularity container to handle GLIBC compatibility:

- **Container**: `verus_server.sif` based on `rust:1.85` Docker image
- **Server**: Flask HTTP API (`server.py`) for Verus code evaluation
- **Client**: Python library (`client.py`) for easy integration

## Bind Mounts

- `./eval_server` → `/server` (server code)
- `${VERUS_PATH}` → `/verus` (Verus binaries)

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Evaluation
```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"code":"use vstd::prelude::*;\nverus! {\nfn test() -> u32 { 42 }\nfn main() {}\n}","timeout":10}'
```

### Batch Evaluation
```bash
curl -X POST http://localhost:5000/evaluate_batch \
  -H "Content-Type: application/json" \
  -d '{"codes":["code1","code2"],"timeout":10}'
```

## Python Client Usage

```python
from eval_server.client import create_client

client = create_client("http://localhost:5000")

if client.health_check():
    print("Server ready!")

stdout, stderr = client.evaluate(verus_code, timeout=10)
if "verified" in stdout:
    print("Verification passed!")
```

## Troubleshooting

**Server won't start:**
```bash
netstat -tuln | grep 5000
pkill -f "singularity run"
```

**Container not found:**
```bash
cd eval_server
rm -f verus_server.sif
singularity build verus_server.sif verus_server.def
```

**Verus not working:**
```bash
singularity exec \
  --bind "${VERUS_PATH}:/verus" \
  verus_server.sif \
  /verus/verus --help
```
