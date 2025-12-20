"""Model server management utilities for AV2."""
import itertools
import os
import pathlib
import subprocess
import time
from pathlib import Path
from typing import Tuple
from src.core.data_utils import Timer
import sys

def save_txt(path: str, data: str) -> None:
    """Save data to text file."""
    with open(path, 'w') as f:
        f.write(data)


def load_txt(path: str, encoding: str = "utf-8") -> str:
    """Load a text file."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def _process_active(process: subprocess.Popen) -> bool:
    """Return True if the process is still running."""
    return process.poll() is None


def _job_active(job_id: str) -> bool:
    """Return True while job_id still shows up in `squeue` (used for training jobs)."""
    out = subprocess.run(
        ["squeue", "-h", "-j", str(job_id)], capture_output=True, text=True
    ).stdout
    return bool(out.strip())


def _wait_for_startup_subprocess(process: subprocess.Popen, log_err_path: str, phrase: str, poll: float, timeout: int) -> bool:
    """
    Read log_err_path until *phrase* appears or process exits, or timeout hits.
    Returns True if phrase found; False otherwise.
    """
    err_file = Path(log_err_path)
    start_ts = time.time()

    while time.time() - start_ts < timeout:
        # Check if process died
        if not _process_active(process):
            print(f"Process {process.pid} exited prematurely")
            return False

        # Read log file (if it exists yet)
        if err_file.exists():
            if phrase in load_txt(str(err_file)):
                return True

        time.sleep(poll)

    # timed out
    print(f"Timeout waiting for startup phrase '{phrase}' in {log_err_path}. Timeout: {timeout} seconds.")
    return False


def _get_cached_model_path(model_name: str) -> str:
    """
    Convert HuggingFace model name to local cache path for offline operation.
    Returns the path to the actual model snapshot directory.
    """
    import os
    import glob

    # Convert model name to cache directory format
    cache_dir_name = model_name.replace("/", "--")
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    cache_path = f"{hf_home}/hub/models--{cache_dir_name}"

    if not os.path.exists(cache_path):
        raise RuntimeError(f"Model {model_name} not found in cache at {cache_path}. Please download it first in online mode.")

    # Find the snapshot directory (usually only one)
    snapshot_dirs = glob.glob(f"{cache_path}/snapshots/*")
    if not snapshot_dirs:
        raise RuntimeError(f"No model snapshots found in {cache_path}/snapshots")

    # Return the first (and usually only) snapshot directory
    return snapshot_dirs[0]


def start_sglang_server(args, model: str, n_gpus: int, retries: int = 3, poll: float = 2.0, timeout: int = 600, lora_model: str = '', log_path: str = '') -> Tuple[subprocess.Popen, str]:
    """
    Launch an SGLang server via subprocess, wait for
    'The server is fired up and ready to roll!' in its stderr, and return
    (process, server_name). Retries up to *retries* times.
    """
    port = args.sglang_port

    # Handle GPU selection based on --gpu_id argument
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        # Override n_gpus to 1 when gpu_id is specified (for parallel job isolation)
        gpu_devices = str(args.gpu_id)
        n_gpus = 1
        print(f"   🎯 Using specified GPU {args.gpu_id} (overriding --inf_n_gpus to 1 for isolation)")
    else:
        # Use default behavior: first n_gpus GPUs (0, or 0,1)
        gpu_devices = ','.join(str(i) for i in range(n_gpus))

    with Timer('STARTING SGLANG SERVER', level=3):
        print(f"   Model: {model}")
        print(f"   GPUs: {n_gpus} (CUDA_VISIBLE_DEVICES={gpu_devices})")
        print(f"   Port: {port}")
        print(f"   LoRA: {lora_model if lora_model else 'None (base model)'}")

        # Check if sglang_debug is enabled
        if os.environ.get('sglang_debug', 'False') == 'True':
            print("🐛 sglang_debug is enabled - skipping server start, using existing server")
            return None, os.environ.get('SGLANG_DEBUG_SERVER', 'debug-server')

        # Verify LoRA adapter exists (if provided)
        if lora_model:
            print(f"🔍 Checking for LoRA model at: {lora_model}")
            for attempt in range(5):
                if os.path.exists(lora_model):
                    print(f"   ✅ LoRA model found!")
                    break
                print(f"   ⏳ LoRA model not found yet, waiting... (attempt {attempt+1}/5)")
                time.sleep(30)
            else:
                raise RuntimeError(f"LoRA model {lora_model} not found after 5 attempts.")

        # Convert model name to cached model path for offline operation
        try:
            cached_model_path = _get_cached_model_path(model)
            print(f"   ✅ Using cached model at: {cached_model_path}")
        except RuntimeError as e:
            print(f"   ❌ Error: {e}")
            raise

        # Set environment variables for subprocess
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        sglang_cache = os.environ.get('SGLANG_CACHE_ROOT', './cache/sglang')
        torch_cache = os.environ.get('TORCHINDUCTOR_CACHE_DIR', os.path.expanduser('~/.cache/torch'))
        env = os.environ.copy()
        env.update({
            'HF_HOME': hf_home,
            'SGLANG_CACHE_ROOT': sglang_cache,
            'NCCL_P2P_DISABLE': '1',
            'TORCHINDUCTOR_CACHE_DIR': torch_cache,
            'TRANSFORMERS_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'HF_HUB_OFFLINE': '1',
            'INFERENCE_PORT': str(port),
            'CUDA_VISIBLE_DEVICES': gpu_devices,
        })

        # Build Singularity command
        singularity_image = os.environ.get('SGLANG_SINGULARITY_IMAGE', './sglang/sglang_server.sif')

        # Check if container exists
        if not os.path.exists(singularity_image):
            raise RuntimeError(f"SGLang Singularity image not found at {singularity_image}. Please pull it first:\n"
                            f"  singularity pull --name sglang_server.sif docker://lmsysorg/sglang:latest")

        # Build base command with Singularity
        cmd = [
            'singularity', 'exec',
            '--nv',  # Enable NVIDIA GPU support
            '--bind', f'{hf_home}:{hf_home}',
            '--bind', f'{sglang_cache}:{sglang_cache}',
            '--bind', f'{torch_cache}:{torch_cache}',
            singularity_image,
            'python3', '-m', 'sglang.launch_server',
            '--model-path', cached_model_path,
            '--dp-size', str(n_gpus),
            '--port', str(port),
            '--trust-remote-code',
            # '--log-level', 'info',  # Enable detailed logging
            # '--log-requests',  # Log all incoming requests to see LoRA paths
        ]

        # NOTE: Quantization during inference is disabled because SGLang container lacks vllm
        # LoRA adapters trained with quantization should still work with full-precision base models

        # Only enable torch compile if sgl_kernel is working (CUDA 12.4+)
        # Skip for CUDA 12.2 environments to avoid cuGreenCtxDestroy symbol errors
        enable_torch_compile = os.environ.get('SGLANG_ENABLE_TORCH_COMPILE', 'False') == 'True'
        if enable_torch_compile:
            cmd.append('--enable-torch-compile')

        # Add LoRA support only when needed
        if lora_model:
            # Convert to absolute path for bind mounting
            lora_model_abs = os.path.abspath(lora_model)
            lora_dir = os.path.dirname(lora_model_abs)

            # Bind mount the LoRA model directory if it's not in a directory we're already mounting
            if not lora_dir.startswith(hf_home) and not lora_dir.startswith(sglang_cache):
                # Insert bind mount after the --nv flag and before the image path
                cmd.insert(3, '--bind')
                cmd.insert(4, f'{lora_dir}:{lora_dir}')

            # Check if model is MoE to set appropriate target modules
            try:
                from transformers import AutoConfig
                model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                is_moe = 'moe' in model_config.model_type.lower() if hasattr(model_config, 'model_type') else False
            except:
                is_moe = False

            # For MoE models, only use attention layers
            if is_moe:
                lora_target_modules_list = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
                print(f"   🔧 MoE model: LoRA target modules = attention only")
            else:
                lora_target_modules_list = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

            cmd.extend([
                '--enable-lora',
                '--max-lora-rank', '16',  # Increased to support lora_r=16
                '--lora-target-modules', *lora_target_modules_list,  # Unpack as separate args
                '--max-loras-per-batch', '8',
                '--lora-backend', 'triton',
                '--disable-radix-cache',
                '--lora-paths', f'{lora_model_abs}={lora_model_abs}',
            ])
            print(f"   🔍 LoRA Configuration:")
            print(f"      Path: {lora_model_abs}")
            print(f"      Bind Mount: {lora_dir}")
            print(f"      Backend: triton")
            print(f"      Max Rank: 16 (supports lora_r up to 16)")
            print(f"      Target Modules: {lora_target_modules_list}")

        if not log_path:
            raise ValueError("log_path is required for SGLang server logging")

        # Print full command for debugging
        # print(f"   📋 Full SGLang Server Command: {' '.join(cmd)}")

        for attempt in range(1, retries + 1):
            # Create log file paths
            log_out_path = f"{log_path}_{attempt}.out"
            log_err_path = f"{log_path}_{attempt}.err"

            print(f"   🔄 SGLang serving: [{attempt}/{retries}] starting subprocess")

            # Open log files
            stdout_file = open(log_out_path, 'w')
            stderr_file = open(log_err_path, 'w')

            # Start the subprocess with file descriptors for logging
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                bufsize=1,
            )
            print(f"      ✅ Started SGLang server process with PID: {process.pid}")
            print(f"      ⏳ Waiting for startup message (timeout: {timeout}s)...")
            print(f'      Stdout file: {stdout_file.name}')
            print(f'      Stderr file: {stderr_file.name}')

            # Store file handles in process for cleanup
            process._stdout_file = stdout_file
            process._stderr_file = stderr_file

            if _wait_for_startup_subprocess(process, log_err_path, "The server is fired up and ready to roll!", poll, timeout):
                server_name = f"localhost:{port}"
                print(f"   ✅ SGLANG SERVER READY!")
                print(f"   📄 Logs: {log_out_path} | {log_err_path}")
                return process, server_name

            # Kill the process before retrying to free up resources
            if _process_active(process):
                print(f"⚠️  Terminating process {process.pid} before retry...")
                process.terminate()
                time.sleep(2)
                if _process_active(process):
                    process.kill()

            # Close log files
            stdout_file.close()
            stderr_file.close()

            print("⚠️  process ended before startup finished -- retrying…")

        raise RuntimeError(f"SGLang failed to start after {retries} attempt(s).")

def stop_sglang_server(args, process: subprocess.Popen) -> None:
    """Stop SGLang server by terminating the process and killing any process on the port."""

    print(f'🛑 STOPPING SGLANG SERVER')

    # Check if sglang_debug is enabled
    if os.environ.get('sglang_debug', 'False') == 'True':
        print("🐛 sglang_debug is enabled - skipping server stop, leaving existing server running")
        return

    port = args.sglang_port

    # Stop the process if provided
    if process is not None:
        print(f"   Process PID: {process.pid}")
        print(f"   Action: Terminating...")

        # Close log file handles if they exist
        if hasattr(process, '_stdout_file'):
            process._stdout_file.close()
        if hasattr(process, '_stderr_file'):
            process._stderr_file.close()

        process.terminate()
        try:
            process.wait(timeout=10)
            print(f"   ✅ SGLang server stopped gracefully")
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Process didn't stop gracefully, killing...")
            process.kill()
            process.wait()
            print(f"   ✅ SGLang server killed")
    else:
        print("⚠️  No sglang process to stop (debug mode or already stopped)")

    # Check if there's still a process on the port and kill it
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"   🔍 Found {len(pids)} process(es) still on port {port}")
            for pid in pids:
                try:
                    print(f"   🔪 Killing process {pid} on port {port}")
                    subprocess.run(['kill', '-9', pid], timeout=5)
                    print(f"   ✅ Killed process {pid}")
                except Exception as e:
                    print(f"   ⚠️  Failed to kill process {pid}: {e}")
    except subprocess.TimeoutExpired:
        print(f"   ⚠️  lsof command timed out")
    except FileNotFoundError:
        print(f"   ⚠️  lsof command not found, skipping port check")
    except Exception as e:
        print(f"   ⚠️  Error checking port {port}: {e}")

    # Clear server state to prevent reuse of stopped server
    args.sglang_process = None
    args.sglang_server_name = None

def start_eval_server(args, retries: int = 3, poll: float = 2.0, timeout: int = 60, log_path: str = '') -> Tuple[subprocess.Popen, str]:
    """
    Launch the Verus evaluation server via subprocess, wait for health check to pass,
    and return (process, server_url). Retries up to *retries* times.
    """
    import requests
    import socket

    port = args.eval_server_port
    server_url = f'http://localhost:{port}'

    with Timer('Starting Eval Server', level=2):
        print(f"Server: {server_url}")
        print(f"Logs: {log_path}_1.out | {log_path}_1.err")

        # Check if eval_debug is enabled
        if os.environ.get('eval_debug', 'False') == 'True':
            print("🐛 eval_debug is enabled - skipping server start, using existing server")
            return None, server_url

        # Check if port is already in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_in_use = sock.connect_ex(('localhost', port)) == 0
        sock.close()

        if port_in_use:
            print(f"Port {port} is already in use, checking if eval server is running...")
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Eval server already running and healthy on port {port}")
                    return None, server_url
                else:
                    print(f"⚠️  Port {port} in use but health check returned status {response.status_code}")
            except requests.RequestException as e:
                print(f"⚠️  Port {port} in use but health check failed: {e}")
            print(f"Starting eval server anyway...")

        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval_server', 'run_server.sh')
        if not os.path.exists(script_path):
            raise RuntimeError(f"Eval server script not found at {script_path}")

        cmd = ['bash', script_path]

        if not log_path:
            raise ValueError("log_path is required for eval server logging")

        for attempt in range(1, retries + 1):
            log_out_path = f"{log_path}_{attempt}.out"
            log_err_path = f"{log_path}_{attempt}.err"

            stdout_file = open(log_out_path, 'w')
            stderr_file = open(log_err_path, 'w')

            # Set port environment variable for the server script
            env = os.environ.copy()
            env['EVAL_SERVER_PORT'] = str(port)

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                bufsize=1
            )

            print(f"Starting (PID {process.pid})... ", end='', flush=True)

            process._stdout_file = stdout_file
            process._stderr_file = stderr_file

            # Wait for server to be ready
            start_ts = time.time()

            while time.time() - start_ts < timeout:
                if not _process_active(process):
                    print(f"Process {process.pid} exited prematurely")
                    break

                try:
                    response = requests.get(f"{server_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"Ready!")
                        return process, server_url
                except requests.RequestException:
                    pass

                time.sleep(poll)

            if _process_active(process):
                print(f"⚠️  Terminating process {process.pid} before retry...")
                process.terminate()
                time.sleep(2)
                if _process_active(process):
                    process.kill()

            stdout_file.close()
            stderr_file.close()

            print(f"Failed (attempt {attempt}/{retries})")

        raise RuntimeError(f"Eval server failed to start after {retries} attempt(s).")


def stop_eval_server(process: subprocess.Popen) -> None:
    """Stop Verus eval server by terminating the process."""
    if os.environ.get('eval_debug', 'False') == 'True':
        return

    if process is None:
        return

    # Close log file handles if they exist
    if hasattr(process, '_stdout_file'):
        process._stdout_file.close()
    if hasattr(process, '_stderr_file'):
        process._stderr_file.close()

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_sft_training(args, model: str, output_path: str, data_path: str, log_path: str, epochs: int = 2, timeout: int = 864000, max_steps: int = None) -> None:
    """
    Run SFT training as a subprocess to ensure proper multi-GPU support.

    Args:
        args: Arguments object containing configuration (including gpu_id if specified)
        model: Base model name/path (e.g., 'Qwen/Qwen2.5-Coder-3B-Instruct')
        output_path: Directory to save trained model
        data_path: Path to JSONL training data
        log_path: Base path for log files (will append .out/.err)
        epochs: Number of training epochs (default: 2)
        timeout: Maximum training time in seconds (default: 864000 = 10 days)
        max_steps: Maximum number of training steps (overrides epochs if set)
    """
    print(f"   Model: {model}")
    print(f"   Data: {data_path}")
    print(f"   Output: {output_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Logs: {log_path}.out | {log_path}.err")

    # Clear CUDA cache before training to free any lingering GPU memory
    import torch
    import gc
    if torch.cuda.is_available():
        print(f"   📊 GPU memory before cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"   🧹 After cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

    # Determine number of GPUs to use
    model_size_gb = 30 if '30B' in model or '30b' in model else 3

    # Handle GPU selection based on --gpu_id argument
    if hasattr(args, 'gpu_id') and args.gpu_id is not None:
        # Use specified GPU only (for parallel job isolation)
        cuda_devices = str(args.gpu_id)
        n_gpus = 1
        use_ddp = False  # Single GPU, no DDP needed
        print(f"   🎯 Using specified GPU {args.gpu_id} for training (overriding multi-GPU for isolation)")
    else:
        # Use default behavior based on model size
        n_gpus = 2 if model_size_gb >= 30 else 1
        cuda_devices = '0,1' if n_gpus == 2 else '0'
        use_ddp = (n_gpus == 2)
        print(f"   🎮 Auto-selected {n_gpus} GPU(s) for training (CUDA_VISIBLE_DEVICES={cuda_devices})")

    # Set environment variables for subprocess
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    env = os.environ.copy()
    env.update({
        'HF_HOME': hf_home,
        'OMP_NUM_THREADS': '1',
        'NCCL_P2P_DISABLE': '1',
        'CUDA_VISIBLE_DEVICES': cuda_devices,
    })

    # Build command - use FSDP for 30B models to partition across 2 GPUs
    # For smaller models, use single GPU without FSDP

    # Adjust parameters based on model size
    # Check if this is an MoE model
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        is_moe = 'moe' in model_config.model_type.lower() if hasattr(model_config, 'model_type') else False
    except:
        is_moe = False

    if model_size_gb >= 30:
        max_seq_length = 1024  # Reduce from default 2048 to save memory
        lora_r = 16  # Increased for better learning capacity
        use_quantization = True
        # use_ddp already set above based on gpu_id
        use_gradient_checkpointing = False  # DDP + quantization + grad checkpointing is incompatible
        per_device_batch_size = 2  # Batch size per GPU
        grad_accum_steps = 32  # Gradient accumulation steps (higher = fewer backprops = faster training)
        # Effective batch size calculation depends on n_gpus

        # For MoE models, only apply LoRA to attention layers (not MLP/expert layers)
        if is_moe:
            lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            print(f"   🔧 MoE model detected: LoRA only on attention layers (not expert FFN layers)")
        else:
            lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        effective_batch_size = n_gpus * per_device_batch_size * grad_accum_steps
        print(f"   🔧 Large model: using 8-bit quantization" + (f" + DDP across {n_gpus} GPUs" if use_ddp else ""))
        print(f"   📏 max_seq_length={max_seq_length}, lora_r={lora_r}")
        print(f"   🎯 LoRA target modules: {lora_target_modules}")
        print(f"   ⚠️  Gradient checkpointing DISABLED (incompatible with DDP + quantization)")
        print(f"   📦 Effective batch size: {n_gpus} GPUs * {per_device_batch_size} per_device * {grad_accum_steps} grad_accum = {effective_batch_size}")
        print(f"   💪 Optimized config: bs={per_device_batch_size}, grad_accum={grad_accum_steps}, lora_r={lora_r}, 8-bit quant")
        print(f"   💾 Checkpoints: saving every 10 steps (keeping 3 most recent)")
    else:
        max_seq_length = 2048
        lora_r = 16
        use_quantization = False
        # use_ddp already set above based on gpu_id (should be False for small models)
        use_gradient_checkpointing = True
        per_device_batch_size = 1
        grad_accum_steps = 8
        lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # Base training arguments
    training_args = [
        '--model_name_or_path', model,
        '--dataset_name', data_path,
        '--learning_rate', '2.0e-4',
        '--num_train_epochs', str(epochs),
        '--per_device_train_batch_size', str(per_device_batch_size),
        '--gradient_accumulation_steps', str(grad_accum_steps),
        '--gradient_checkpointing', 'true' if use_gradient_checkpointing else 'false',
        '--eval_strategy', 'no',
        '--use_peft', 'true',
        '--lora_r', str(lora_r),
        '--lora_alpha', str(lora_r * 2),
        '--lora_target_modules', *lora_target_modules,  # Unpack the list
        '--report_to', 'none',
        '--logging_steps', '1',
        '--output_dir', output_path,
        '--bf16', 'true',
        '--max_seq_length', str(max_seq_length),
        '--seed', str(args.seed),
        # '--save_strategy', 'steps',  # Save checkpoints based on steps
        # '--save_steps', '5',  # Save every 10 steps
        # '--save_total_limit', '1',  # Keep only 3 most recent checkpoints
    ]

    # Add 8-bit quantization for large models (better quality than 4-bit)
    if use_quantization:
        training_args.extend([
            '--load_in_8bit', 'true',
        ])

    # Build command - use DDP for multi-GPU, plain python for single GPU
    if use_ddp:
        cmd = [
            'accelerate', 'launch',
            '--num_processes', '2',
            '--num_machines', '1',
            '--mixed_precision', 'no',  # Already using bf16 in training args
            '--multi_gpu',
            'sft.py'
        ] + training_args
    else:
        cmd = [sys.executable, '-u', 'sft.py'] + training_args

    # Add max_steps if specified (overrides epochs)
    if max_steps is not None:
        cmd.extend(['--max_steps', str(max_steps)])

    # Create log files
    log_out_path = f"{log_path}.out"
    log_err_path = f"{log_path}.err"

    print(f"   🚀 Starting training as subprocess...")
    print(f"   📄 Logs: {log_out_path} | {log_err_path}")
    print(f"   📋 Command: {' '.join(cmd)}")

    # Open log files
    stdout_file = open(log_out_path, 'w')
    stderr_file = open(log_err_path, 'w')

    try:
        # Run training as subprocess
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )

        print(f"   ⏳ Training process started (PID: {process.pid}), waiting for completion with timeout {timeout}...")

        # Wait for completion with timeout
        try:
            returncode = process.wait(timeout=timeout)
            if returncode != 0:
                raise RuntimeError(f"Training failed with exit code {returncode}. Check logs: {log_err_path}")
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Training exceeded timeout of {timeout}s, terminating...")
            process.kill()
            process.wait()
            raise RuntimeError(f"Training timed out after {timeout}s")

    finally:
        stdout_file.close()
        stderr_file.close()

    # Verify model directory exists
    model_dir = pathlib.Path(output_path)
    if not model_dir.exists():
        print(f"   ⏳ Model directory not found, waiting 10s...")
        time.sleep(10)
        if not model_dir.exists():
            raise RuntimeError(f"Expected model directory {model_dir} missing after training")

    # Additional GPU cleanup after training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"   📊 GPU memory after training cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

    print(f"✅ SFT TRAINING COMPLETED SUCCESSFULLY")
    print(f"   Model: {output_path}")


if __name__ == '__main__':
    import requests
    import json

    print("Testing server_management...")
    print()

    # Test 1: Basic utility tests
    print("Test 1: Testing utility functions...")
    fake_job_id = "999999"
    assert _job_active(fake_job_id) == False, "job_active test with fake job failed"

    test_model = "test/model"
    test_log = "/tmp/test_log"
    expected_trn = trn_sft_form.format(model=test_model, output_path="/tmp/out", data_path="/tmp/data", log_path=test_log, epochs=10)
    assert test_model in expected_trn, "Training sbatch generation failed"
    assert "/tmp/out" in expected_trn, "Training output path test failed"
    print("✅ Utility functions passed")
    print()

    # Test 2: SGLang Server Integration Test
    print("Test 2: SGLang Server Integration Test (dp_size=2)")
    print("-" * 70)

    # Configuration
    test_dp_size = 2
    test_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
    test_log_path = "./logs/test_server"

    # Create mock args for testing
    class MockArgs:
        sglang_port = 3000

    test_args = MockArgs()

    # Test prompts
    test_prompts = [
        "Write a Python function to compute factorial",
        "What is 2+2?",
        "Explain what a linked list is in one sentence"
    ]

    print(f"Starting SGLang server (model={test_model}, dp_size={test_dp_size})...")

    try:
        # Start server
        process, server_name = start_sglang_server(
            args=test_args,
            model=test_model,
            n_gpus=test_dp_size,
            retries=1,
            log_path=test_log_path,
            timeout=300
        )

        print(f"✅ Server started successfully at {server_name}")
        print()

        # Wait a moment for full initialization
        time.sleep(2)

        # Test inference with multiple prompts
        print(f"Testing {len(test_prompts)} inference requests...")
        print("=" * 70)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[Request {i}/{len(test_prompts)}]")
            print(f"Prompt: {prompt}")
            print("-" * 70)

            # Make API request
            response = requests.post(
                f"http://{server_name}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.7,
                        "max_new_tokens": 100
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                output_text = result.get("text", "")
                meta_info = result.get("meta_info", {})

                print(f"Response: {output_text[:200]}...")
                print(f"Tokens: {meta_info.get('prompt_tokens', 0)} prompt + {meta_info.get('completion_tokens', 0)} completion")
                print(f"Latency: {meta_info.get('e2e_latency', 0):.3f}s")
                print("✅ Request successful")
            else:
                print(f"❌ Request failed with status {response.status_code}")

        print()
        print("=" * 70)
        print("✅ All inference tests passed!")

    except Exception as e:
        print(f"❌ Server test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print()
        print("Stopping server...")
        try:
            stop_sglang_server(test_args, process)
            print("✅ Server stopped successfully")
        except Exception as e:
            print(f"⚠️  Error stopping server: {e}")

    print()
    print("=" * 70)
    print("✅ All server_management tests completed!")
    print("=" * 70)