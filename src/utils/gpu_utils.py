"""GPU utilities for physical GPU detection and mapping."""

import os
import re
from pathlib import Path
from typing import Optional


def get_physical_gpu_index(logical_gpu_id: int = 0) -> int:
    """
    Map a logical GPU index (as seen by CUDA_VISIBLE_DEVICES) to its physical GPU index.

    This is crucial for SLURM environments where CUDA_VISIBLE_DEVICES might be set to
    a subset of GPUs (e.g., "4,6"), making logical GPU 0 correspond to physical GPU 4.

    Args:
        logical_gpu_id: Logical GPU index (default 0, the first visible GPU)

    Returns:
        Physical GPU index in the system. Returns 0 if detection fails.

    Example:
        If SLURM sets CUDA_VISIBLE_DEVICES="4,6":
        - get_physical_gpu_index(0) -> 4 (first visible GPU is physical GPU 4)
        - get_physical_gpu_index(1) -> 6 (second visible GPU is physical GPU 6)
    """
    try:
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetPciInfo,
            nvmlShutdown
        )

        # Initialize NVML
        nvmlInit()

        try:
            # Get PCI bus ID for the logical GPU
            handle = nvmlDeviceGetHandleByIndex(logical_gpu_id)
            pci_info = nvmlDeviceGetPciInfo(handle)
            target_bus_id = pci_info.busId.decode() if hasattr(pci_info.busId, "decode") else pci_info.busId

            # Read ALL physical GPUs from /proc/driver/nvidia/gpus/
            # This bypasses all SLURM/CUDA restrictions
            gpu_proc_dir = Path('/proc/driver/nvidia/gpus')
            all_physical_gpus = []

            if gpu_proc_dir.exists():
                for gpu_dir in sorted(gpu_proc_dir.iterdir()):
                    if gpu_dir.is_dir():
                        # Read the information file
                        info_file = gpu_dir / 'information'
                        if info_file.exists():
                            with open(info_file, 'r') as f:
                                content = f.read()
                                # Extract PCI bus ID - format is like "0000:17:00.0"
                                bus_match = re.search(r'Bus Location:\s+([0-9a-fA-F:\.]+)', content)
                                if bus_match:
                                    pci_bus = bus_match.group(1)
                                    all_physical_gpus.append(pci_bus)

            # Sort to get consistent physical ordering
            all_physical_gpus.sort()

            # Normalize PCI bus IDs for comparison
            def normalize_pci(pci_id: str) -> str:
                """Convert PCI ID to comparable format (remove leading zeros, make lowercase)."""
                parts = pci_id.replace('0x', '').lower().split(':')
                if len(parts) >= 2:
                    # Handle both "0000:17:00.0" and "00000000:17:00.0" formats
                    return ':'.join(parts[-3:]) if len(parts) >= 3 else ':'.join(parts[-2:])
                return pci_id.lower()

            target_normalized = normalize_pci(target_bus_id)

            # Find our GPU in the sorted list
            physical_index = None
            for idx, pci in enumerate(all_physical_gpus):
                normalized = normalize_pci(pci)
                if normalized == target_normalized or target_normalized in normalized or normalized in target_normalized:
                    physical_index = idx
                    break

            return physical_index if physical_index is not None else 0

        finally:
            # Always cleanup NVML
            nvmlShutdown()

    except Exception as e:
        # If anything fails (no GPU, pynvml not available, etc.), default to 0
        print(f"⚠️  Warning: Could not detect physical GPU index: {e}")
        print(f"⚠️  Defaulting to physical_gpu_index=0 (no port offset)")
        return 0


def get_gpu_mapping_info(logical_gpu_id: int = 0) -> dict:
    """
    Get detailed GPU mapping information for logging/debugging.

    Args:
        logical_gpu_id: Logical GPU index

    Returns:
        Dictionary with keys: logical_id, physical_id, cuda_visible_devices, pci_bus_id
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    physical_id = get_physical_gpu_index(logical_gpu_id)

    return {
        'logical_id': logical_gpu_id,
        'physical_id': physical_id,
        'cuda_visible_devices': cuda_visible,
    }
