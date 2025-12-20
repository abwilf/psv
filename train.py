import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import sys
from src.core.prompt_utils import get_inf_prompt, get_spec_completion_pairs
from src.core.evaluation import run_codes, get_eval_metrics, passed, compute_pass_at, add_to_metrics, check_eval_server_health
from src.core.run_cache import create_run_cache, RunMetadata, ModelMetadata
from src.core.io_utils import save_jsonl, save_pk, load_pk, txt_debug, generate_model_name_for_cache
from src.core.server_management import stop_sglang_server, run_sft_training
from src.core.data_utils import Timer, create_seed_ds
import os
import time
from src.proposal.train_proposer import train_proposer
from inference import inference, inference_and_eval
import argparse
import subprocess
import os
from os.path import exists, join
import wandb
import torch

def set_seed(seed):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For reproducibility in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_filtered_training_data(ds, args):
    """Get training data filtered by ablation strategy (if applicable).

    Returns filtered dataframe of solved, non-test questions. If filtertrainsols
    ablation is active, filters to problems with pass_rate between hardthresh and medthresh.
    """
    # Get base training data (solved, non-test questions)
    train_data = ds.loc[ds.solved & ~ds.is_test_dataset]

    # Apply filtertrainsols ablation if specified
    if args.ablation == 'filtertrainsols':
        # Parse thresholds from proposal_strategy
        strategy_parts = args.proposal_strategy.split('_')
        relevant_parts = strategy_parts[1:strategy_parts.index('ablation')] if 'ablation' in strategy_parts else strategy_parts[1:]
        cfg = {k:v for k,v in zip(relevant_parts[::2], relevant_parts[1::2])}
        hardthresh = float(cfg['hardthresh'])
        medthresh = float(cfg['medthresh'])

        # Filter to problems with pass_rate between hardthresh and medthresh
        original_len = len(train_data)
        train_data = train_data.loc[(train_data['pass_rate'] >= hardthresh) &
                                     (train_data['pass_rate'] <= medthresh)]
        print(f"🔧 filtertrainsols: Filtered from {original_len} to {len(train_data)} problems (pass_rate ∈ [{hardthresh}, {medthresh}])")

    return train_data

def main(args):
    print("\n" + "="*80)
    print("🎯 STUBBED MAIN - PRINTING ARGS:")
    print("="*80)
    for key, value in sorted(vars(args).items()):
        print(f"  {key:30s} = {value}")
    print("="*80 + "\n")
    # return  # Exit early for testing

    solver_model, proposal_model = args.model, args.model
    ds = create_seed_ds(args)
    is_alphaverus = 'alphaverus' in args.name.lower()

    for iteration in tqdm(range(args.n_iterations), desc=f'Iterations'):
        with Timer(f"Iteration {iteration} / {args.n_iterations-1}", level=1):
            args.iteration = iteration

            ## Inference and eval, add to ds
            # Skip for AlphaVerus (will use final eval with final_inf_k instead)
            if not is_alphaverus:
                with Timer("Inference and Evaluation", level=2):
                    ds, metrics = inference_and_eval(args, ds, iteration, solver_model, is_alphaverus)
                    ds_path = join(args.ds_basepath, f'iter{iteration}.jsonl')
                    ds.to_json(ds_path, orient='records', lines=True)

                    # Add dataset info to metrics
                    metrics['ds_len'] = len(ds)
                    metrics['len_trn_data'] = len(get_filtered_training_data(ds, args))

                    # Add proposal stats from previous iteration if available
                    if hasattr(args, 'proposal_stats') and args.proposal_stats is not None:
                        print(f'Has proposal stats!!!')
                        metrics.update(args.proposal_stats)
                    else:
                        print(f'No proposal stats!!!')

                    wandb.log(metrics, step=iteration)

                    # For noadapt ablation: save baseline pass rates in memory after iteration 0
                    if iteration == 0 and args.ablation == 'noadapt':
                        args.baseline_pass_rates = {row['question_id']: row['pass_rate'] for _, row in ds.iterrows() if 'pass_rate' in row and pd.notna(row['pass_rate'])}
                        print(f"💾 Saved {len(args.baseline_pass_rates)} baseline pass rates in memory for noadapt ablation")

                    # Save base model metadata at iteration 0
                    if iteration == 0:
                        base_model_metadata = ModelMetadata(
                            iteration=0,
                            model_type='base',
                            base_model_path=args.model,
                            lora_path=None,
                            model_cache_name=generate_model_name_for_cache(args.model, 0, proposal_strategy=args.proposal_strategy, train_dataset=args.train_dataset, epochs=args.epochs, inf_fs=args.inf_fs, inf_k=args.inf_k),
                            training_args=None,
                            training_dataset_size=None,
                            training_time_seconds=None,
                            timestamp=datetime.now().isoformat()
                        )
                        args.cache.save_model_metadata(base_model_metadata)
                        print(f'✅ Saved base model metadata for iteration 0')

                    if args.n_iterations==1:
                        break
            
                ## SFT -> model_{t+1}
                with Timer("SFT Training", level=2):
                    # Check if model already exists in cache
                    if args.cache.model_exists(iteration + 1):
                        model_path = str(args.cache.get_model_path(iteration + 1))
                        print(f'✅ Model already exists at: {model_path}')
                    else:
                        # Generate temporary path for training
                        model_cache_name = generate_model_name_for_cache(args.model, iteration+1, args.proposal_strategy, args.max_n_qs, args.train_dataset, args.epochs, args.trained_proposer, inf_fs=args.inf_fs, inf_k=args.inf_k)
                        temp_model_path = os.path.join(args.output_dir, 'temp_model', model_cache_name)
                        model_path = temp_model_path
                        print(f'🆕 Creating new model -> {model_path}')
                        stop_sglang_server(args, args.sglang_process)

                        # Create SFT dataset
                        if args.skip_sol_verification:
                            print(f"🔧 -solutionver ablation: Training on one solution per question (passing or failing)")
                            # Get all non-test questions
                            train_solved = ds.loc[~ds.is_test_dataset]
                            train_solved['trainable_solution'] = train_solved.apply(lambda row: random.choice([row['passing_solution'], row['failing_solution']]) if row['passing_solution'] else row['failing_solution'], axis=1)
                        else:
                            train_solved = get_filtered_training_data(ds, args)
                            train_solved['trainable_solution'] = train_solved['passing_solution']
                            print(f"📚 Creating SFT data from {len(train_solved)} solved questions (excluding test datasets)")

                        # Build SFT data with proper spec-completion pairs for each example
                        data = []
                        for elt in train_solved.to_dict('records'):
                            data.append({
                                'prompt': get_inf_prompt(spec_in=elt['spec'], k=0, spec_completion_pairs=[]),
                                'completion': [{'role': 'assistant', 'content': f"```rust\n{elt['trainable_solution']}\n```\n"}]
                            })
                        results = run_codes([elt['spec']+'\n'+elt['trainable_solution'] for elt in train_solved.to_dict('records') ])
                        if not args.skip_sol_verification:
                            assert all(results)

                        print(f'📝 SFT dataset size: {len(data)} examples')
                        sft_data_path = join(args.sft_basepath, f'iter{iteration}.jsonl')
                        save_jsonl(sft_data_path, data)

                        # Run SFT training as subprocess
                        training_start = time.time()
                        run_sft_training(
                            args=args,
                            model=args.model,
                            output_path=model_path,
                            data_path=sft_data_path,
                            log_path=join(args.trn_log_basepath, f'iter{iteration}'),
                            epochs=args.epochs,
                            max_steps=args.trn_max_steps,
                            timeout=args.trn_timeout
                        )
                        training_time = time.time() - training_start

                        # Save model to cache
                        model_metadata = ModelMetadata(
                            iteration=iteration + 1,
                            model_type='lora',
                            base_model_path=args.model,
                            lora_path=None,  # Will be set after copying
                            model_cache_name=model_cache_name,
                            training_args={
                                'epochs': args.epochs,
                                'max_steps': args.trn_max_steps,
                                'train_dataset': args.train_dataset,
                            },
                            training_dataset_size=len(data),
                            training_time_seconds=training_time,
                            timestamp=datetime.now().isoformat()
                        )
                        args.cache.save_model_from_path(model_path, iteration + 1, model_metadata)

                        # Update model_path to cached location
                        model_path = str(args.cache.get_model_path(iteration + 1))

                    solver_model = model_path
                    print(f"🎯 Current model for next iteration: {solver_model}")

                ## Problem proposal (for next round)
                if iteration < args.n_iterations-1:
                    with Timer(f'Problem proposal', level=2):
                        if args.trained_proposer and iteration >= 2:
                            print(f"🎯 Using trained proposer model!")
                            args.proposal_model = train_proposer(args, iteration, ds)
                        else:
                            args.proposal_model = args.model
                            print(f"🎯 Using base model for proposals: {proposal_model}")

                        # For noadapt ablation: restore baseline pass rates from memory before problem proposal
                        if args.ablation == 'noadapt' and iteration > 0:
                            assert hasattr(args, 'baseline_pass_rates') and args.baseline_pass_rates
                            for qid, rate in args.baseline_pass_rates.items():
                                ds.loc[ds['question_id'] == qid, 'pass_rate'] = rate
                                print(f"🔄 Restored {len(args.baseline_pass_rates)} baseline pass rates from memory for noadapt ablation")

                        if 'rft' not in args.proposal_strategy:
                            # Check if proposed questions are already cached
                            proposal_cache = args.cache.load_proposed_questions(iteration)

                            if proposal_cache is not None:
                                print(f'✅ Found cached proposed questions for iteration {iteration}')
                                # Reconstruct ds_new from cached questions
                                deduplicated = proposal_cache.get_final_questions()
                                print(f'📊 Loaded {len(deduplicated)} cached proposed questions')

                                to_add = {
                                    'nl_desc_oai': [q.nl_desc for q in deduplicated],
                                    'spec': [q.question_spec for q in deduplicated],
                                    'task': [q.dataset for q in deduplicated],
                                    'task_id': [f"synth_{iteration}_{i}_{q.question_id[:32]}" for i, q in enumerate(deduplicated)],
                                    'solved': [False] * len(deduplicated),
                                    'parent': [('|'.join(q.parents) if q.parents else '') for q in deduplicated],
                                    'ancestor': [''] * len(deduplicated),
                                    'proposal_eligible': [True] * len(deduplicated),
                                    'hard_proposed': [False] * len(deduplicated),
                                    'is_train_dataset': [False] * len(deduplicated),
                                    'is_test_dataset': [False] * len(deduplicated),
                                    'qtype': [q.proposal_strategy for q in deduplicated],
                                    'question_id': [q.question_id for q in deduplicated],
                                }
                                ds_new = pd.DataFrame(to_add)

                                # Calculate stats from cache
                                if args.proposal_strategy.startswith('icl_'):
                                    n_generated = len(proposal_cache.all_generated)
                                    n_valid = len(proposal_cache.valid)
                                    n_unique = len(proposal_cache.deduplicated)
                                    uniqueness_rate = n_unique / n_valid if n_valid > 0 else 0.0

                                    print(f"📊 Cached proposal stats - Generated: {n_generated}, Valid: {n_valid}, Unique: {n_unique}, Uniqueness rate: {uniqueness_rate:.2%}")
                                    args.proposal_stats = {
                                        'proposal/n_generated': n_generated,
                                        'proposal/n_valid': n_valid,
                                        'proposal/n_unique': n_unique,
                                        'proposal/uniqueness_rate': uniqueness_rate,
                                        'proposal/valid_rate': n_valid / n_generated if n_generated > 0 else 0.0,
                                    }
                                else:
                                    args.proposal_stats = None
                            else:
                                # No cache found - generate new proposals
                                print(f'📭 No cached proposed questions found for iteration {iteration}, generating new ones...')

                                # Dynamically load the appropriate proposal strategy
                                if args.proposal_strategy.startswith('icl_'):
                                    from src.proposal.strategies.icl_band import icl_band_strat as prob_proposal
                                    ds_new, proposal_stats = prob_proposal(ds, args, iteration=iteration)
                                    print(f"📊 Proposal stats - Generated: {proposal_stats['n_generated']}, Valid: {proposal_stats['n_valid']}, Unique: {proposal_stats['n_unique']}, Uniqueness rate: {proposal_stats['uniqueness_rate']:.2%}")
                                    # Store proposal stats for logging in next iteration
                                    args.proposal_stats = {
                                        'proposal/n_generated': proposal_stats['n_generated'],
                                        'proposal/n_valid': proposal_stats['n_valid'],
                                        'proposal/n_unique': proposal_stats['n_unique'],
                                        'proposal/uniqueness_rate': proposal_stats['uniqueness_rate'],
                                        'proposal/valid_rate': proposal_stats['n_valid'] / proposal_stats['n_generated'] if proposal_stats['n_generated'] > 0 else 0.0,
                                    }
                                else:
                                    raise ValueError(f"Unknown proposal strategy: {args.proposal_strategy}")

                                # Save proposal cache if it was created
                                if hasattr(args, 'proposal_cache') and args.proposal_cache is not None:
                                    args.cache.save_proposed_questions(args.proposal_cache)
                                    print(f'✅ Saved proposal cache for iteration {args.proposal_cache.iteration}')
                                    args.proposal_cache = None  # Clear for next iteration

                            ds = pd.concat([ds, ds_new])
                            ds.reset_index(drop=True, inplace=True)
                        else:
                            args.proposal_stats = None  # No stats for rft strategy
                            print(f"Skipping problem proposal b/c proposal strategy is {args.proposal_strategy}")

            ## Final eval (if last iter)
            if iteration == args.n_iterations - 1:
                with Timer('Final Eval', level=2):
                    # Use final_inf_k for better pass@k estimates with separate cache
                    args.inf_k = args.final_inf_k
                    print(f"🎯 Final evaluation using inf_k={args.final_inf_k} for better pass@k estimates")

                    # Filter out synthetic datasets for final evaluation
                    ds_filtered = ds.loc[ds.is_train_dataset | ds.is_test_dataset]
                    ds_filtered, metrics = inference_and_eval(args, ds_filtered, iteration+1, solver_model, is_alphaverus)

                    # Add proposal stats from previous iteration if available
                    if hasattr(args, 'proposal_stats') and args.proposal_stats is not None:
                        metrics.update(args.proposal_stats)

                    wandb.log(metrics, step=iteration+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='dafny2verus', help='Dataset to use for training and question evolution')
    parser.add_argument('--test_datasets', type=str, default='mbpp,humaneval', help='Comma-separated list of datasets to test on')
    parser.add_argument('--n_iterations', type=int, default=5, help='Num examples to give in Few Shot')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Coder-3B-Instruct', help='Model name/path to use')
    parser.add_argument('--run_dir_base', type=str, default='./outputs', help='Base directory for all experiment runs')
    parser.add_argument('--skip_cache', action='store_true', help='Skip reading from all caches (inferences, questions, models) but still write to them')
    parser.add_argument('--eval_server_mode', type=str, default='local', choices=['local', 'remote'], help='Eval server mode: local (localhost) or remote (SSH tunnel)')
    parser.add_argument('--eval_server_host', type=str, default='localhost', help='Remote host for eval server (only used if eval_server_mode=remote)')

    # Inference
    parser.add_argument('--inf_k', type=int, default=5, help='For pass@k')
    parser.add_argument('--final_inf_k', type=int, default=100, help='k for final evaluation pass@k (larger for better estimates)')
    parser.add_argument('--inf_temp', type=float, default=0.8, help='For inference')
    parser.add_argument('--inf_top_p', type=float, default=0.9, help='')
    parser.add_argument('--inf_max_tokens', type=int, default=2048, help='')
    parser.add_argument('--inf_fs', type=int, default=1, help='Num examples to give in Few Shot for solving')
    parser.add_argument('--inf_n_gpus', type=int, default=1, help='N gpus to use for SGLANG serving at inference time')
    parser.add_argument('--inf_bs', type=int, default=2000, help='')
    parser.add_argument('--inf_client_timeout', type=int, default=36000, help='Client timeout (s) before cutting off the request to inference (includes queuing!). Default: 10 hours.')
    parser.add_argument('--inf_port', type=int, default=3002, help='Port for inference server')
    parser.add_argument('--sglang_cache_root', type=str, default='./cache/sglang', help='SGLang cache root directory')
    parser.add_argument('--sglang_inference_port', type=int, default=3002, help='SGLang inference port')
    parser.add_argument('--spec_k', type=int, default=2, help='Pass@k for spec validation during problem proposal')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use for both inference and training (e.g., 0 or 1). If not set, defaults to GPU 0 for single-GPU operations and GPUs 0,1 for multi-GPU operations.')

    # Question gen
    parser.add_argument('--proposal_strategy', type=str, default='icl_input_all_output_hard_fsk_30_easythresh_0.8_medthresh_0.6_hardthresh_0.2', help='')
    parser.add_argument('--max_n_qs', type=int, default=50, help='Max number of questions to have per iteration per strategy')
    parser.add_argument('--qgen_mult', type=int, default=25, help='Need to run this * number of questions asked for b/c some fail or are duplicates')

    # SFT
    parser.add_argument('--trn_fs_k', type=int, default=0, help='Num examples to give in Few Shot as part of training (increases input context length)')
    parser.add_argument('--trn_max_steps', type=int, default=None, help='Num examples to give in Few Shot as part of training (increases input context length)')
    parser.add_argument('--trn_timeout', type=int, default=28800, help='Training timeout in seconds (default: 28800 = 8 hours)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs for SFT')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Other
    parser.add_argument('--tags', type=str, default='mitm', help='Tags for wandb')
    parser.add_argument('--name', type=str, default='h2e_scaling1', help='Name for wandb run')

    # Accept --config from wandb agent but ignore it (we use wandb.config instead)
    parser.add_argument('--config', type=str, default=None, help='Wandb sweep config (ignored, use wandb.config instead)')

    args = parser.parse_args()

    # Read required ports from environment variables (fail if not set)
    args.sglang_port = int(os.environ['SGLANG_PORT'])
    args.eval_server_port = int(os.environ['EVAL_SERVER_PORT'])
    print(f"🔌 SGLang port from env: {args.sglang_port}")
    print(f"🔌 Eval server port from env: {args.eval_server_port}")

    # Set random seeds for reproducibility
    set_seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")

    # Set up environment variables for compatibility with existing code
    os.environ['INFERENCE_PORT'] = str(args.sglang_inference_port)
    os.environ['SGLANG_CACHE_ROOT'] = args.sglang_cache_root
    os.environ['sglang_debug'] = 'False'
    os.environ['DEBUG_MODE'] = 'False'
    args.job_id = os.environ.get("SLURM_JOB_ID")

    # Initialize wandb without name first (will set it after getting config)
    args.wandb_run = wandb.init(project="simple8", entity="socialiq", config=vars(args), tags=args.tags.split(','))

    # Override args with wandb config if it exists (for sweep support)
    if hasattr(wandb.config, 'config') and wandb.config.config:
        print("🔄 Overriding args with wandb sweep config:")
        for key, value in wandb.config.config.items():
            old_value = getattr(args, key, None)
            setattr(args, key, value)
            print(f"  {key}: {old_value} -> {value}")

    # Set the run name to match args.name (after config override)
    wandb.run.name = args.name + '-seed' + str(args.seed)
    wandb.run.tags = args.tags.split(',')
    print(f"📝 W&B run name set to: {args.name}")

    args.output_dir = os.path.join(args.run_dir_base, args.wandb_run.name)
    args.trained_proposer = 'trained_proposer' in args.proposal_strategy
    args.debug_mode = 'debug' in args.tags
    assert not args.trained_proposer, 'Not implemented'

    # Parse ablation parameter from proposal_strategy (keep original string intact for caching)
    # Format: icl_input_X_output_Y_fsk_Z_easythresh_A_medthresh_B_hardthresh_C_ablation_TYPE
    strategy_parts = args.proposal_strategy.split('_')
    if 'ablation' in strategy_parts:
        ablation_idx = strategy_parts.index('ablation')
        args.ablation = strategy_parts[ablation_idx + 1] if ablation_idx + 1 < len(strategy_parts) else 'none'
    else:
        args.ablation = 'none'  # Default to standard baseline

    print(f"🔧 Ablation mode: {args.ablation}")

    # Parse verification ablation flags
    args.skip_spec_verification = '-specver' in args.ablation
    args.skip_sol_verification = '-solutionver' in args.ablation
    print(f"🔧 Skip spec verification: {args.skip_spec_verification}")
    print(f"🔧 Skip solution verification: {args.skip_sol_verification}")
    if args.skip_spec_verification:
        args.qgen_mult = 1

    args.difficulty_blind = 'nodiff' in args.proposal_strategy
    print(f"Difficulty aware problem proposal: {args.difficulty_blind}")

    # Parse use_human_data from proposal_strategy
    # Format options: ...humandata (uses human/gold data) or ...zerodata (uses generated passing solutions)
    # Default: use generated data (False)
    if 'humandata' in strategy_parts:
        args.inf_use_human_data = True
    elif 'zerodata' in strategy_parts:
        args.inf_use_human_data = False
    else:
        args.inf_use_human_data = False  # Default to generated data

    print(f"🔧 Use human data for few-shot: {args.inf_use_human_data}")

    # Validate ablation type
    valid_ablations = ['none', 'nodiff', 'nosampling', 'noadapt', 'samplingbaseonly', 'noheader', '-specver', '-solutionver', '-specver-solutionver', 'filtertrainsols']
    if args.ablation not in valid_ablations:
        raise ValueError(f"Invalid ablation type '{args.ablation}'. Must be one of: {valid_ablations}")
    
    # set up cache dirs for interrupted runs to continue with --name flag
    dir_map = {
        "ds_basepath": "ds",
        "inf_basepath": "inf_resp",
        "sft_basepath": "sft_data",
        "trn_log_basepath": "trn_logs",
        "sglang_log_basepath": "sglang_logs",
        "nl_desc_basepath": f"nl_desc",
        "spec_basepath": f"spec_{args.proposal_strategy}",
    }

    for attr, subdir in dir_map.items():
        path = join(args.output_dir, subdir)
        setattr(args, attr, path)
        os.makedirs(path, exist_ok=True)
    
    # debugging: if you don't want to restart the server uncomment the second line
    args.sglang_process, args.sglang_server_name = None, None
    # args.sglang_process, args.sglang_server_name = None, 'localhost:3000'

    # Init cache
    args.cache = create_run_cache(
        run_name=args.wandb_run.name,
        skip_cache=args.skip_cache,
        base_dir=args.run_dir_base
    )

    # Save run metadata
    from datetime import datetime
    run_metadata = RunMetadata(
        run_name=args.wandb_run.name,
        run_dir=str(args.cache.run_dir),
        args=vars(args),
        start_time=datetime.now().isoformat(),
        last_updated=datetime.now().isoformat(),
        n_iterations_completed=0
    )
    args.cache.save_run_metadata(run_metadata)

    print(f"📁 Run directory: {args.cache.run_dir}")
    print(f"🔧 Skip cache (read-only): {args.skip_cache}")

    # Check eval server health
    check_eval_server_health(args)

    try:
        main(args)
    finally:
        print(f'Stopping sglang server before exiting...')
        stop_sglang_server(args, args.sglang_process)
        # print(f'✅ Keeping SGLang server running (manual management mode)')
