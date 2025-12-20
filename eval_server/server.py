#!/usr/bin/env python3
"""
Verus evaluation server - runs inside Singularity container.

Provides HTTP API for executing Verus verification on code snippets.
"""

import os
import subprocess
import tempfile
import uuid
import time
import glob
import multiprocessing as mp
from typing import Tuple, List
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

VERUS_ROOT = os.environ.get("VERUS_ROOT", "/verus")
VERUS_PATH = os.path.join(VERUS_ROOT, "verus")
TEMP_DIR = "/scratch/rust_files"

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================================
# Core evaluation functions (ported from evaluation.py)
# ============================================================================

def run_code(file_name: str, timeout_duration: int = 10) -> Tuple[str, str]:
    """Run Verus on a single file."""
    try:
        result = subprocess.run(
            [VERUS_PATH, file_name, '--multiple-errors', '100'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_duration
        )
        return result.stdout.decode('utf-8'), result.stderr.decode('utf-8')
    except subprocess.TimeoutExpired:
        return "", f"Process timed out after {timeout_duration} seconds"


def run_code_wrapper(code: str, timeout_duration: int = 10, i: int = 0) -> Tuple[str, str]:
    """Wrapper for run_code with retry logic for panics."""
    if i == 5:
        return '', ''

    # Create a temporary file
    temp_file = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.rs")

    try:
        with open(temp_file, "w") as f:
            f.write(code)

        # Run the code
        result = run_code(temp_file, timeout_duration)

        # Retry if panic occurs
        if "thread '<unnamed>' panicked" in result[1]:
            print(f'Panic retry {i}')
            time.sleep(5)
            result = run_code_wrapper(code, timeout_duration, i + 1)

        return result
    finally:
        # Delete the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _run_indexed(job: Tuple[int, str, int]) -> Tuple[int, bool, str, str]:
    """Worker function for parallel code evaluation."""
    idx, code, timeout = job
    out, err = run_code_wrapper(code, timeout)
    success = out != ''
    return idx, success, out, err


def eval_codes_noserver(codes: List[str], timeout: int = 10, quiet: bool = False) -> List[Tuple[bool, str, str]]:
    """
    Parallel evaluation with an in-order result list.
    Returns [(success, stdout, stderr), …] in the same order as `codes`.
    """
    import sys
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    n = len(codes)
    results: List[Tuple[bool, str, str] | None] = [None] * n

    try:
        with mp.Pool(processes=30) as pool:
            jobs = [(idx, code, timeout) for idx, code in enumerate(codes)]

            if has_tqdm:
                # Progress bar writes to stderr
                with tqdm(total=n, desc="Evaluating", ncols=80, disable=quiet, file=sys.stderr) as pbar:
                    for idx, success, out, err in pool.imap_unordered(_run_indexed, jobs):
                        results[idx] = (success, out, err)
                        pbar.update(1)
            else:
                # No progress bar
                for idx, success, out, err in pool.imap_unordered(_run_indexed, jobs):
                    results[idx] = (success, out, err)

    except OSError as e:
        if "No space left on device" in str(e):
            print(f"Multiprocessing failed, falling back to sequential...", file=sys.stderr)
            # Sequential fallback
            for idx, code in enumerate(codes):
                idx_result, success, out, err = _run_indexed((idx, code, timeout))
                results[idx] = (success, out, err)
        else:
            raise

    # Cleanup
    for pattern in ("*.long-type*", "dcgm-gpu*"):
        for p in glob.glob(pattern):
            try:
                os.remove(p)
            except:
                pass

    return results


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "verus_path": VERUS_PATH})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate a single Verus code snippet.

    Request JSON:
        {
            "code": "verus code string",
            "timeout": 10  # optional, seconds
        }

    Response JSON:
        {
            "success": true/false,
            "stdout": "...",
            "stderr": "...",
            "timed_out": true/false
        }
    """
    try:
        data = request.get_json()
        code = data.get('code')
        timeout = data.get('timeout', 10)

        if not code:
            return jsonify({"error": "No code provided"}), 400

        # Create temporary file
        temp_file = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.rs")

        try:
            with open(temp_file, 'w') as f:
                f.write(code)

            # Run Verus
            result = subprocess.run(
                [VERUS_PATH, temp_file, '--multiple-errors', '100'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout
            )

            response = {
                "success": True,
                "stdout": result.stdout.decode('utf-8', errors='replace'),
                "stderr": result.stderr.decode('utf-8', errors='replace'),
                "timed_out": False,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            response = {
                "success": False,
                "stdout": "",
                "stderr": f"Process timed out after {timeout} seconds",
                "timed_out": True,
                "return_code": -1
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate_batch', methods=['POST'])
def evaluate_batch():
    """
    Evaluate multiple Verus code snippets in parallel.

    Request JSON:
        {
            "codes": ["code1", "code2", ...],
            "timeout": 10  # optional, seconds per code
        }

    Response JSON:
        {
            "results": [
                {"success": true, "stdout": "...", "stderr": "...", "timed_out": false},
                ...
            ]
        }
    """
    try:
        data = request.get_json()
        codes = data.get('codes', [])
        timeout = data.get('timeout', 10)

        if not codes:
            return jsonify({"error": "No codes provided"}), 400

        # Use parallel evaluation
        eval_results = eval_codes_noserver(codes, timeout=timeout, quiet=True)

        # Convert to response format
        results = []
        for success, stdout, stderr in eval_results:
            timed_out = "timed out" in stderr.lower()
            results.append({
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
                "return_code": 0 if success else -1
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('EVAL_SERVER_PORT', 5000))
    print(f"Starting Verus evaluation server on port {port}")
    print(f"Verus binary: {VERUS_PATH}")
    print(f"Temp directory: {TEMP_DIR}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
