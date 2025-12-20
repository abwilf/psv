"""
Per-run caching system for AV2.

Each experiment run has its own directory with isolated caches for:
- Inferences (per iteration)
- Proposed questions (per iteration)
- Models (per iteration)

All cache operations are atomic: load all or compute all and save all.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
from datetime import datetime


@dataclass
class InferenceResult:
    """Complete inference result for a single question."""
    question_id: str
    question_spec: str
    dataset: str
    nl_desc: str
    gold_code: str
    model_name: str
    pass_rate: float
    n_passing: int
    n_total: int
    passing_solution: str
    failing_solution: str
    all_solutions: List[str]
    pass_at_k: Dict[str, float]  # {'pass@1': 0.5, 'pass@2': 0.75, ...}
    timestamp: str

    # Additional metadata for analysis
    task_id: Optional[str] = None
    parent: Optional[str] = None
    ancestor: Optional[str] = None
    is_train_dataset: bool = False
    is_test_dataset: bool = False


@dataclass
class InferenceCache:
    """Cache for all inference results in an iteration."""
    iteration: int
    model_name: str
    results: List[InferenceResult]
    metadata: Dict[str, Any]  # inference args like k, fs, etc.
    timestamp: str

    def to_dict(self) -> Dict[str, InferenceResult]:
        """Convert to dict keyed by question_id for easy lookup."""
        return {r.question_id: r for r in self.results}


@dataclass
class ProposedQuestion:
    """Complete information about a proposed question."""
    question_id: str
    question_spec: str
    nl_desc: str
    proposal_strategy: str
    proposal_model: str
    dataset: str
    parents: List[str]
    iteration: int
    base_ds: str
    gold_code: str

    # Validation results
    is_valid: bool  # passed spec verification
    is_duplicate: bool
    validation_error: Optional[str] = None

    # Generation metadata (for analysis)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    # Inference results (populated after inference runs)
    was_solved: Optional[bool] = None
    pass_rate: Optional[float] = None
    n_passing: Optional[int] = None
    n_total: Optional[int] = None
    passing_solution: Optional[str] = None
    all_solutions: Optional[List[str]] = None
    pass_at_k: Optional[Dict[str, float]] = None


@dataclass
class ProposedQuestionsCache:
    """Cache for all proposed questions in an iteration."""
    iteration: int
    proposal_strategy: str
    proposal_model: str

    # Store all questions at different stages for analysis
    all_generated: List[ProposedQuestion]  # all generated questions
    valid: List[ProposedQuestion]  # passed spec verification
    deduplicated: List[ProposedQuestion]  # final set after dedup

    metadata: Dict[str, Any]  # proposal args, stats, etc.
    timestamp: str

    def get_final_questions(self) -> List[ProposedQuestion]:
        """Get the final deduplicated questions."""
        return self.deduplicated


@dataclass
class ModelMetadata:
    """Metadata about a model at a specific iteration."""
    iteration: int
    model_type: str  # 'base' or 'lora'
    base_model_path: str
    lora_path: Optional[str]  # path to adapter in models/iterN/
    model_cache_name: str  # the cache name used for inference
    timestamp: str

    # Training information (for LoRA models)
    training_args: Optional[Dict[str, Any]] = None
    training_dataset_size: Optional[int] = None
    training_time_seconds: Optional[float] = None


@dataclass
class RunMetadata:
    """Metadata about the entire run."""
    run_name: str
    run_dir: str
    args: Dict[str, Any]  # all command-line args
    start_time: str
    last_updated: str
    n_iterations_completed: int


class RunCache:
    """
    Manages all caching for a single experiment run.

    Provides atomic read/write operations:
    - If cache exists and not skip_cache: load it
    - If cache doesn't exist or skip_cache: compute and save it
    - skip_cache only affects reading; writing always happens
    """

    def __init__(self, run_dir: str, skip_cache: bool = False):
        """
        Initialize run cache.

        Args:
            run_dir: Path to run directory (e.g., ./outputs/{run_name})
            skip_cache: If True, skip reading from caches (but still write to them)
        """
        self.run_dir = Path(run_dir)
        self.skip_cache = skip_cache

        # Create directory structure (always needed since we always write)
        self.models_dir = self.run_dir / "models"
        self.inferences_dir = self.run_dir / "inferences"
        self.proposed_questions_dir = self.run_dir / "proposed_questions"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.inferences_dir.mkdir(parents=True, exist_ok=True)
        self.proposed_questions_dir.mkdir(parents=True, exist_ok=True)

    # ==================== Inference Cache ====================

    def get_inference_cache_path(self, iteration: int) -> Path:
        """Get path to inference cache file for an iteration."""
        return self.inferences_dir / f"iter{iteration}.pkl"

    def load_inferences(self, iteration: int) -> Optional[InferenceCache]:
        """
        Load cached inference results for an iteration.

        Returns:
            InferenceCache if exists and not skip_cache, None otherwise
        """
        if self.skip_cache:
            return None

        cache_path = self.get_inference_cache_path(iteration)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load inference cache from {cache_path}: {e}")
            return None

    def save_inferences(self, cache: InferenceCache):
        """
        Save inference results for an iteration.

        Atomic write: saves to temp file then moves.
        """
        cache_path = self.get_inference_cache_path(cache.iteration)
        temp_path = cache_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(cache_path)
            print(f"Saved inference cache to {cache_path}")
        except Exception as e:
            print(f"Error: Failed to save inference cache to {cache_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    # ==================== Proposed Questions Cache ====================

    def get_proposed_questions_cache_path(self, iteration: int) -> Path:
        """Get path to proposed questions cache file for an iteration."""
        return self.proposed_questions_dir / f"iter{iteration}.pkl"

    def load_proposed_questions(self, iteration: int) -> Optional[ProposedQuestionsCache]:
        """
        Load cached proposed questions for an iteration.

        Returns:
            ProposedQuestionsCache if exists and not skip_cache, None otherwise
        """
        if self.skip_cache:
            return None

        cache_path = self.get_proposed_questions_cache_path(iteration)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load proposed questions cache from {cache_path}: {e}")
            return None

    def save_proposed_questions(self, cache: ProposedQuestionsCache):
        """
        Save proposed questions for an iteration.

        Atomic write: saves to temp file then moves.
        """
        cache_path = self.get_proposed_questions_cache_path(cache.iteration)
        temp_path = cache_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(cache_path)
            print(f"Saved proposed questions cache to {cache_path}")
        except Exception as e:
            print(f"Error: Failed to save proposed questions cache to {cache_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    # ==================== Model Cache ====================

    def get_model_metadata_path(self, iteration: int) -> Path:
        """Get path to model metadata file for an iteration."""
        return self.models_dir / f"iter{iteration}_metadata.pkl"

    def get_model_path(self, iteration: int) -> Path:
        """
        Get path to model directory for an iteration.

        For iter 0 (base model), returns the base model path from metadata.
        For iter 1+, returns the LoRA adapter directory.
        """
        if iteration == 0:
            # Base model - return metadata path, actual model path in metadata
            return self.get_model_metadata_path(iteration)
        else:
            # LoRA adapter directory
            return self.models_dir / f"iter{iteration}"

    def load_model_metadata(self, iteration: int) -> Optional[ModelMetadata]:
        """
        Load model metadata for an iteration.

        Returns:
            ModelMetadata if exists and not skip_cache, None otherwise
        """
        if self.skip_cache:
            return None

        metadata_path = self.get_model_metadata_path(iteration)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load model metadata from {metadata_path}: {e}")
            return None

    def save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata for an iteration."""
        metadata_path = self.get_model_metadata_path(metadata.iteration)
        temp_path = metadata_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(metadata_path)
            print(f"Saved model metadata to {metadata_path}")
        except Exception as e:
            print(f"Error: Failed to save model metadata to {metadata_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def model_exists(self, iteration: int) -> bool:
        """
        Check if model exists for an iteration.

        For iter 0: checks if metadata exists
        For iter 1+: checks if LoRA directory exists and has adapter files
        """
        if self.skip_cache:
            return False

        if iteration == 0:
            return self.get_model_metadata_path(iteration).exists()
        else:
            model_dir = self.get_model_path(iteration)
            if not model_dir.exists():
                return False
            # Check for essential adapter files
            adapter_model = model_dir / "adapter_model.safetensors"
            adapter_config = model_dir / "adapter_config.json"
            return adapter_model.exists() and adapter_config.exists()

    def save_model_from_path(self, source_path: str, iteration: int, metadata: ModelMetadata):
        """
        Save a trained model by copying from source path.

        Args:
            source_path: Path to trained model directory (for LoRA) or model name (for base)
            iteration: Iteration number
            metadata: Model metadata to save
        """
        # Save metadata
        self.save_model_metadata(metadata)

        # For iter 0, just save metadata (base model path is in metadata)
        if iteration == 0:
            return

        # For iter 1+, copy LoRA adapter files
        target_dir = self.get_model_path(iteration)
        if target_dir.exists():
            print(f"Warning: Model directory {target_dir} already exists, overwriting...")
            shutil.rmtree(target_dir)

        try:
            shutil.copytree(source_path, target_dir)
            print(f"Saved model to {target_dir}")
        except Exception as e:
            print(f"Error: Failed to copy model from {source_path} to {target_dir}: {e}")

    # ==================== Run Metadata ====================

    def get_run_metadata_path(self) -> Path:
        """Get path to run metadata file."""
        return self.run_dir / "run_metadata.pkl"

    def load_run_metadata(self) -> Optional[RunMetadata]:
        """Load run metadata."""
        if self.skip_cache:
            return None

        metadata_path = self.get_run_metadata_path()
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load run metadata from {metadata_path}: {e}")
            return None

    def save_run_metadata(self, metadata: RunMetadata):
        """Save run metadata."""
        metadata_path = self.get_run_metadata_path()
        temp_path = metadata_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(metadata_path)
        except Exception as e:
            print(f"Error: Failed to save run metadata to {metadata_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def update_run_metadata(self, n_iterations_completed: int):
        """Update run metadata with latest iteration count.

        Note: Run metadata updates always work, even with skip_cache enabled,
        since metadata tracks run state rather than cached results.
        """
        # Temporarily disable skip_cache for metadata updates
        original_skip_cache = self.skip_cache
        self.skip_cache = False

        metadata = self.load_run_metadata()
        if metadata:
            metadata.n_iterations_completed = n_iterations_completed
            metadata.last_updated = datetime.now().isoformat()
            self.save_run_metadata(metadata)

        # Restore original skip_cache setting
        self.skip_cache = original_skip_cache


def create_run_cache(run_name: str, skip_cache: bool = False, base_dir: str = "./outputs") -> RunCache:
    """
    Create a RunCache for a given run name.

    Args:
        run_name: Name of the run (from WandB)
        skip_cache: If True, skip reading from caches (but still write to them)
        base_dir: Base directory for all runs

    Returns:
        RunCache instance
    """
    run_dir = os.path.join(base_dir, run_name)
    return RunCache(run_dir, skip_cache=skip_cache)
