"""ICL band problem proposal strategy.

This strategy generates new verification problems using in-context learning with
difficulty-based categorization and adaptive pass rate thresholds.

Ablations (controlled via ablation parameter in strategy name):
- none (baseline): Standard ICLBand with all features
  * Difficulty categorization (easy/medium/hard based on pass rates)
  * Adaptive (recompute pass rates each iteration)
  * Random sampling (different examples each prompt)
  * Sample from base + synthetic questions

- nodiff: No difficulty categorization
  * Tests if difficulty labels matter or if diverse examples suffice
  * Treats all problems equally (no easy/medium/hard labels in prompt)
  * Still adaptive and samples from base + synthetic

- nosampling: Fixed prompt + base only + no difficulty
  * Tests simplest approach: fixed prompt with base dataset
  * Fixed prompt (sample once, reuse for all queries)
  * Only base dataset (no synthetic in prompt)
  * No difficulty categorization

- noadapt: No adaptivity
  * Tests if adaptive difficulty assessment matters
  * Freezes pass rates from iteration 0
  * Still has difficulty categorization and random sampling

- samplingbaseonly: Base dataset only in prompt
  * Tests if synthetic questions in prompt help
  * Only base dataset (no synthetic in prompt)
  * Still adaptive with difficulty categorization
"""
import random
import numpy as np
from os.path import exists, join
from typing import Tuple, List
from tqdm import tqdm
import pandas as pd
import copy
from src.core.data_utils import flatten, Timer
from src.core.io_utils import txt_debug
from src.core.prompt_utils import extract_rust_block
from src.core.evaluation import run_specs, clean_spec
from src.core.problem_utils import get_last_fn_name
from src.core.inference_utils import inference_and_passk, make_spec_pass_fn
from src.core.run_cache import ProposedQuestion, ProposedQuestionsCache
from inference import inference
import uuid
import hashlib
from datetime import datetime

# Multi easy-to-hard specific prompt
icl_band_base_prompt_spec = \
'''I would like you to output a function spec in Verus (Rust) that is of difficulty {difficulty}. A spec defines a function's inputs, outputs, and returns, and may define preconditions (requires) and postconditions (ensures) as well as any helpers necessary for defining these attributes.

First you should reason about your answer, then you should output the problem descriptions for each problem within ```rust ``` tags, so that it can be parsed.

Your solution will take the form:

# Reasoning
< your observations about what makes the example problems easy or hard, and ideas about how to propose a new problem >

```rust
<function spec you propose>
```

Here are some examples of specs{diff_aware_part1}.

{examples}

Now it's your turn! Please{diff_aware_part2} output a problem that is **{difficulty}** for the model by either making problems that are not challenging enough harder, or making problems that are too challenging easier, or both, in some creative combination.  Please enclose your function spec in ```rust ``` tags, so that it can be parsed. DO NOT copy any of the examples, and DO NOT include any function implementations. You should END your target function with an open curly brace {{.
'''

difficulty_aware_part1 = ' and how difficult they were for the model'
difficulty_aware_part2 = ' describe what you think makes these problems easy, medium, hard, or impossible, then'

def sample_fewshot_examples(ds: pd.DataFrame, categories: List[str], fsk: int) -> List[Tuple[str, List, List[str]]]:
    """
    Sample few-shot examples evenly across categories with min() check to avoid oversampling.

    Args:
        ds: DataFrame with problems and 'problem_class' column
        categories: List of problem difficulty categories to sample from
        fsk: Total number of few-shot examples to sample

    Returns:
        List of tuples (category, sampled_problems, question_ids)
    """
    budget_per_category = fsk // len(categories)
    sampled_examples = []

    for category in categories:
        available = ds.loc[ds['problem_class'] == category]
        n_to_sample = min(len(available), budget_per_category)

        if n_to_sample > 0:
            sampled = available.sample(n_to_sample).to_dict('records')
            question_ids = [p.get('question_id', '') for p in sampled]
            sampled_examples.append((category, sampled, question_ids))
        else:
            global warning_printed
            if not warning_printed:
                print(f"⚠️  Warning: No {category} examples available to sample")
                warning_printed = True

    return sampled_examples


def build_fewshot_prompt(sampled_examples: List[Tuple[str, List, List[str]]], skip_headers: bool = False, include_difficulty: bool = False) -> Tuple[str, List[str]]:
    """
    Build few-shot prompt string from sampled examples.

    Args:
        sampled_examples: List of tuples (category, problems, question_ids)
        skip_headers: If True, skip difficulty category headers AND shuffle examples
                     to prevent difficulty information leaking through ordering

    Returns:
        Tuple of (prompt_string, all_question_ids)
    """
    prompt_parts = []
    all_question_ids = []

    if skip_headers:
        # When skipping headers, flatten all examples and shuffle to prevent
        # difficulty information from leaking through ordering
        all_examples = []
        for category, problems, question_ids in sampled_examples:
            for i, problem in enumerate(problems):
                all_examples.append((category, problem, question_ids[i]))

        # Shuffle to remove any difficulty ordering
        random.shuffle(all_examples)

        # Build prompt with shuffled examples
        for idx, (category, problem, qid) in enumerate(all_examples, 1):
            if include_difficulty:
                prompt_parts.append(f'**Example {idx}: {category.upper()} **\n```rust\n{problem["spec"]}\n```\n')
            else:
                prompt_parts.append(f'**Example {idx}**\n```rust\n{problem["spec"]}\n```\n')
            all_question_ids.append(qid)
    else:
        # Standard: keep difficulty categories separated and labeled
        for category, problems, question_ids in sampled_examples:
            prompt_parts.append(f'-- {category.upper()} Examples --')

            for i, problem in enumerate(problems):
                prompt_parts.append(f'**{category.upper()} Problem {i+1}**\n```rust\n{problem["spec"]}\n```\n')
            all_question_ids.extend(question_ids)

    return '\n'.join(prompt_parts), all_question_ids

nosample_examples = None
def icl_band_strat(ds, args, iteration) -> Tuple[pd.DataFrame, dict]:
    """
    ICL band strategy: generate problems using in-context learning with band-based difficulty progression.

    Strategy name format: icl_{input}_{output}_fsk_{number}
    - input: which difficulty examples to show {all, ai, easy, medium, hard, impossible}
      - 'all': sample from easy, medium, hard (excludes impossible)
      - 'ai': sample from easy, medium, hard, impossible (includes all difficulties)
      - specific difficulty: sample only from that difficulty level
    - output: which difficulty to generate {uniform, ui, easy, medium, hard, impossible}
      - 'uniform': randomly select from easy, medium, hard (excludes impossible)
      - 'ui': randomly select from easy, medium, hard, impossible (includes all difficulties)
      - specific difficulty: generate problems at that difficulty level
    - fsk_{number}: total number of few-shot examples, split evenly across input categories

    Examples:
      - icl_input_all_output_hard_fsk_30: Sample 30 examples (10 from each of 3 difficulties), generate hard problems
      - icl_input_easy_output_hard_fsk_30: Sample 30 easy examples, generate hard problems
      - icl_input_all_output_uniform_fsk_20: Sample 20 examples (~7 from each of 3 difficulties), randomly select output difficulty
      - icl_input_ai_output_ui_fsk_30: Sample 30 examples (~7-8 from each of 4 difficulties including impossible), randomly select output difficulty including impossible

    Args:
        ds: Current dataset
        args: Command line arguments
        iteration: Current iteration number

    Returns:
        Tuple[pd.DataFrame, dict]:
            - DataFrame with new proposed problems
            - Stats dict with keys: n_generated, n_valid, n_unique, uniqueness_rate
    """
    global warning_printed
    warning_printed = False

    print(f"🎯 ICL BAND STRATEGY - ABLATION: {args.ablation.upper()}")

    # Don't propose or consider any test questions
    ds = ds.loc[~ds.is_test_dataset]

    # Filter dataset based on ablation type
    if args.ablation in ['nosampling', 'samplingbaseonly', 'noadapt']:
        ds = ds.loc[ds.is_train_dataset]
        print(f"[Ablation]: Filtering to base train dataset only ({len(ds)} problems)")

    # Parse strategy name to get input/output configuration. Format: icl_input_{X}_output_{Y}_fsk_{Z}_easythresh_{A}_medthresh_{B}_hardthresh_{C}_ablation_{TYPE}
    strategy_parts = args.proposal_strategy.split('_')
    relevant_parts = strategy_parts[1:strategy_parts.index('ablation')] if 'ablation' in strategy_parts else strategy_parts[1:]
    cfg = {k:v for k,v in zip(relevant_parts[::2], relevant_parts[1::2])}

    # Initialize collections for all passing specs
    all_passing_specs = []
    all_passing_parents = []
    existing_fn_names = set()  # Start with empty set (ignore cache for dedup)
    existing_specs = set()     # Start with empty set (ignore cache for dedup)
    fsk = int(cfg['fsk'])
    assert cfg['output'] in ['easy', 'medium', 'hard', 'uniform', 'ui']

    # Metrics tracking
    total_generated = args.max_n_qs
    total_valid = 0
    total_unique = 0

    # Generate exactly max_n_qs inferences (no multiplier, no loop)
    n_infs_needed = args.max_n_qs
    print(f'📊 Generating {n_infs_needed} inferences...')

    # Get NL Descs
    all_msgs, _parents, _ancestors, _types, _target_diffs = [], [], [], [], []

    # Difficulty aware stuff
    if args.difficulty_blind: # just used solved problems
        ds['problem_class'] = 'impossible'
        ds.loc[ds['pass_rate'] > 0, 'problem_class'] = 'solved'
        input_categories = ['solved']
    else:
        ds['problem_class'] = 'impossible'
        ds.loc[ds['pass_rate'] >= float(cfg['hardthresh']), 'problem_class'] = 'hard'
        ds.loc[ds['pass_rate'] >= float(cfg['medthresh']), 'problem_class'] = 'medium'
        ds.loc[ds['pass_rate'] >= float(cfg['easythresh']), 'problem_class'] = 'easy'
        # Determine input categories based on strategy
        if cfg['input'] == 'all':
            input_categories = ['easy', 'medium', 'hard']
        elif cfg['input'] == 'ai':
            input_categories = ['easy', 'medium', 'hard', 'impossible']
        else:
            input_categories = [cfg['input']]

    for _ in range(n_infs_needed):
        # Sample few-shot examples using helper function
        if args.ablation=='nosampling':
            global nosample_examples, sampled_question_ids
            if nosample_examples is None:
                print(f'\n-- Sampling examples for problem generation. This should only happen once because the ablation is nosampling! --\n')
                sampled_examples = sample_fewshot_examples(ds, ['easy', 'medium', 'hard', 'impossible'], fsk)
                nosample_examples, sampled_question_ids = build_fewshot_prompt(sampled_examples, skip_headers=True)
        else:
            sampled_examples = sample_fewshot_examples(ds, input_categories, fsk)
            fs_examples, sampled_question_ids = build_fewshot_prompt(sampled_examples, skip_headers=args.difficulty_blind or 'noheader' in args.ablation, include_difficulty='noheader' in args.ablation)

        # Determine target difficulty for generation using helper function
        if cfg['output'] == 'uniform':
            target_diff = random.choice(['easy', 'medium', 'hard'])
        elif cfg['output'] == 'ui':
            target_diff = random.choice(['easy', 'medium', 'hard', 'impossible'])
        else:
            target_diff = cfg['output']
        msgs = [
            {'role': 'system', 'content': f'You are an exceptionally strong Verus (Rust) programmer.'},
            {'role': 'user', 'content': icl_band_base_prompt_spec.format(
                examples=nosample_examples if args.ablation=='nosampling' else fs_examples,
                difficulty=target_diff.upper(),
                diff_aware_part1='' if args.difficulty_blind else difficulty_aware_part1,
                diff_aware_part2='' if args.difficulty_blind else difficulty_aware_part2,
            )},
        ]
        all_msgs.append(msgs)
        _parents.append('|'.join([qid for qid in sampled_question_ids if qid]))
        _ancestors.append('')
        _types.append(args.proposal_strategy + f'_target_{target_diff}')
        _target_diffs.append(target_diff)

    # Run inference and evaluation
    pass_fn = make_spec_pass_fn(skip_verification=args.skip_spec_verification) # Create pass function for spec evaluation
    responses, solved = inference_and_passk(
        all_msgs=all_msgs,
        sampling_params={"model": args.model, "temperature": 0.8, "max_tokens": 1024, "timeout": args.inf_client_timeout},
        question_idxs=np.arange(n_infs_needed),
        pass_fn=pass_fn,
        args=args,
        inf_bs=args.inf_bs,
        return_format='tuple',
    )
    passing_idxs = np.where(solved)[0]
    passing_specs = np.array(responses)[passing_idxs]
    passing_parents = np.array(_parents)[passing_idxs]

    # Update valid count
    total_valid = len(passing_specs)
    print(f'✅ Validation complete: {total_valid} / {total_generated} specs passed = {100*total_valid/total_generated:.2f}%')

    # dedup
    for elt,parent in zip(passing_specs, passing_parents):
        # Use clean_spec to get cleaned version (with verus! wrapper and imports if needed)
        cleaned_spec = clean_spec(elt)
        last_fn_name = get_last_fn_name(elt)

        # Dedup on both spec content and function name
        if (cleaned_spec not in existing_specs and last_fn_name not in existing_fn_names):
            all_passing_specs.append(cleaned_spec)
            all_passing_parents.append(parent)
            existing_specs.add(cleaned_spec)
            existing_fn_names.add(last_fn_name)

    # Update unique count
    total_unique = len(all_passing_specs)
    uniqueness_rate = total_unique / total_valid if total_valid > 0 else 0.0
    print(f'🎯 After deduplication: {total_unique} unique specs (uniqueness rate: {100*uniqueness_rate:.2f}%)')

    passing_specs = all_passing_specs
    passing_parents = all_passing_parents
    passing_questions = [''] * len(passing_specs)
    passing_ancestors = [''] * len(passing_specs)
    passing_types = [args.proposal_strategy] * len(passing_specs)

    # Track all questions (generated, valid, deduplicated)
    all_generated_questions = []
    valid_questions = []
    deduplicated_questions = []

    # Store all generated questions (both passing and failing validation)
    for i, (msg, parent) in enumerate(zip(all_msgs, _parents)):
        spec_raw = responses[i] if i < len(responses) else ""
        is_valid = i in passing_idxs

        # Generate question_id (only for valid specs for consistency)
        if is_valid:
            # Use spec content and dataset for deterministic ID
            spec_for_hash = spec_raw
            qid_str = f"{spec_for_hash}_{args.train_dataset}"
            question_id = hashlib.sha256(qid_str.encode()).hexdigest()
        else:
            # For invalid specs, use a random ID
            question_id = str(uuid.uuid4())

        question = ProposedQuestion(
            question_id=question_id,
            question_spec=spec_raw,
            nl_desc='',
            proposal_strategy=args.proposal_strategy,
            proposal_model=args.proposal_model,
            dataset=f'synthetic-iter={iteration}',
            parents=parent.split('|') if parent else [],
            iteration=iteration,
            base_ds=args.train_dataset,
            gold_code='',
            is_valid=is_valid,
            is_duplicate=False,  # Will be set below for valid questions
            validation_error=None if is_valid else "Failed spec verification",
            generation_metadata={'target_difficulty': _target_diffs[i]}
        )
        all_generated_questions.append(question)
        if is_valid:
            valid_questions.append(question)

    # Mark duplicates in valid questions
    seen_specs = set()
    seen_fn_names = set()
    passing_question_ids = []

    for question in valid_questions:
        cleaned = clean_spec(question.question_spec)
        fn_name = get_last_fn_name(question.question_spec)

        if cleaned not in seen_specs and fn_name not in seen_fn_names:
            question.is_duplicate = False
            deduplicated_questions.append(question)
            seen_specs.add(cleaned)
            seen_fn_names.add(fn_name)
            passing_question_ids.append(question.question_id)
        else:
            question.is_duplicate = True

    # Create ProposedQuestionsCache object
    proposal_cache = ProposedQuestionsCache(
        iteration=iteration,
        proposal_strategy=args.proposal_strategy,
        proposal_model=args.proposal_model,
        all_generated=all_generated_questions,
        valid=valid_questions,
        deduplicated=deduplicated_questions,
        metadata={
            'fsk': fsk,
            'max_n_qs': args.max_n_qs,
            'target_difficulties': cfg.get('output', 'uniform'),
            'ablation': args.ablation,
        },
        timestamp=datetime.now().isoformat()
    )

    # Attach to args for train.py to save
    args.proposal_cache = proposal_cache

    print(f'📦 Created proposal cache with {len(all_generated_questions)} generated, {len(valid_questions)} valid, {len(deduplicated_questions)} deduplicated questions')

    assert len(set(passing_question_ids))==len(passing_question_ids), "Duplicate question IDs detected"

    # Return only the newly generated specs (no cache concatenation)
    to_ret = (
        np.array(passing_specs),
        np.array(passing_questions),
        np.array(passing_parents),
        np.array(passing_ancestors),
        np.array(passing_types),
        np.array(passing_question_ids) if len(passing_specs) > 0 else np.array([]),
    )

    # Returning
    specs, questions, parents, ancestors, types, question_ids = to_ret
    unique_task_ids = [f"synth_{iteration}_{i}_{str(uuid.uuid4())[:32]}" for i in range(len(specs))]

    to_add = {
        'nl_desc_oai': questions,
        'spec': specs,
        'task': f'synthetic-iter={iteration}',
        'task_id': unique_task_ids,  # Add unique task_ids
        'solved': False,
        'parent': parents,
        'ancestor': ancestors,
        'proposal_eligible': True,
        'hard_proposed': False,
        'is_train_dataset': False,
        'is_test_dataset': False,
        'qtype': types,
        'question_id': question_ids,
    }
    df = pd.DataFrame(to_add)

    # Build stats dict
    stats = {
        'n_generated': total_generated,
        'n_valid': total_valid,
        'n_unique': total_unique,
        'uniqueness_rate': uniqueness_rate
    }

    return df, stats

