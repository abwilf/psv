import openai
import asyncio
from pprint import pprint
from src.core.prompt_utils import get_inf_prompt, postprocess_verus, get_spec_completion_pairs
from src.core.io_utils import save_pk, load_pk, txt_debug, generate_model_name_for_cache
from src.core.server_management import start_sglang_server, stop_sglang_server
from src.core.evaluation import add_to_metrics, run_codes, K_LIST
from src.core.run_cache import InferenceResult, InferenceCache
from src.core.data_utils import Timer
from src.core.inference_utils import inference_and_passk, make_completion_pass_fn, estimate_pass_at_k
import os
import random
# Helper function to get inference port from args or environment
def get_inference_port(args=None):
    if args and hasattr(args, 'inference_port'):
        return args.inference_port
    return int(os.environ.get('INFERENCE_PORT', '3000'))
from os.path import join, exists
import os
from sshtunnel import SSHTunnelForwarder
import asyncssh
from tqdm import tqdm
from httpx import TimeoutException
import numpy as np

SSH_KEY = os.path.expanduser('~/.ssh/id_rsa')
client = None

def debug_inference(all_msgs, sampling_params):
    """Generate fake responses with random words for debug mode."""
    words = ['apple', 'banana', 'cat', 'dog', 'elephant', 'fish', 'grape', 'house',
             'ice', 'jump', 'kite', 'lion', 'moon', 'nest', 'orange', 'piano']

    class Message:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, message):
            self.message = message

    class Response:
        def __init__(self, messages):
            self.choices = [Choice(Message(content)) for content in messages]

    results = []
    n = sampling_params.get("n", 1)
    for _ in all_msgs:
        responses = [' '.join(random.sample(words, k=min(5, len(words)))) for _ in range(n)]
        results.append(Response(responses))

    return results

def merge_cached_and_new_results(specs, cached_responses, cached_solved, cached_sol_at_k, cached_passed, cached_question_ids,
                                uncached_specs, new_responses, new_solved, new_sol_at_k, new_passed, k, args):
    """Merge cached results with new inference results in the correct order."""
    # Create mapping from spec to its index in original specs list
    spec_to_index = {spec: i for i, spec in enumerate(specs)}
    uncached_spec_to_index = {spec: i for i, spec in enumerate(uncached_specs)}

    # Initialize result arrays
    responses = [""] * len(specs)
    solved = [False] * len(specs)
    passed = [[False] * k for _ in range(len(specs))]
    sol_at_k = cached_sol_at_k.copy()

    # Fill in cached results (they should be in order for the cached specs)
    cached_idx = 0
    for i, spec in enumerate(specs):
        if spec not in uncached_specs and cached_idx < len(cached_responses):
            responses[i] = cached_responses[cached_idx]
            solved[i] = cached_solved[cached_idx]
            if cached_idx < len(cached_passed) and len(cached_passed[cached_idx]) > 0:
                passed[i] = cached_passed[cached_idx][:k] + [False] * max(0, k - len(cached_passed[cached_idx]))
            cached_idx += 1

    # Fill in new results
    for spec in uncached_specs:
        if spec in spec_to_index and spec in uncached_spec_to_index:
            orig_idx = spec_to_index[spec]
            new_idx = uncached_spec_to_index[spec]

            responses[orig_idx] = new_responses[new_idx]
            solved[orig_idx] = new_solved[new_idx]
            passed[orig_idx] = new_passed[new_idx].tolist()

            # Add to sol_at_k
            for k_str, values in new_sol_at_k.items():
                if k_str not in sol_at_k:
                    sol_at_k[k_str] = [False] * len(specs)
                sol_at_k[k_str][orig_idx] = values[new_idx]

    print(f"🔄 Merged {cached_idx} cached + {len(uncached_specs)} new results = {len(specs)} total")

    return responses, solved, sol_at_k, np.array(passed)

async def fetch_completion(client, msgs, sampling_params):
    try:
        return await client.chat.completions.create(
            messages=msgs,
            **sampling_params,
        )
    except Exception as e:
        print(f"Inference Error: {e}")
        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, message):
                self.message = message

        class Response:
            def __init__(self, messages):
                # messages is a list of strings
                self.choices = [Choice(Message(content)) for content in messages]
                
        return Response([""]*sampling_params["n"])
        
client = None
async def inference_async(all_msgs, sampling_params, sglang_server_name, max_concurrency=500, args=None):
    # Get timeout from args or use default (10 hours)
    client_timeout = getattr(args, 'inf_client_timeout', 36000) if args else 36000

    # Check if server is localhost - if so, connect directly without SSH
    if sglang_server_name.startswith('localhost:'):
        # Extract port from server name
        port = int(sglang_server_name.split(':')[1])
        print(f"    🔗 Connecting directly to local server at localhost:{port}")

        # Connect directly to localhost without SSH tunneling
        # Use a long timeout to prevent client-side cancellations during long inference
        async with openai.AsyncOpenAI(
            api_key='EMPTY',
            base_url=f'http://localhost:{port}/v1',
            timeout=float(client_timeout)
        ) as client:
            sem = asyncio.Semaphore(max_concurrency)

            async def run_one(msgs):
                async with sem:
                    return await fetch_completion(client, msgs, sampling_params)

            tasks = [asyncio.create_task(run_one(msgs)) for msgs in all_msgs]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return results
    else:
        # Remote server - use SSH tunneling
        print(f"    🔗 Connecting to remote server via SSH: {sglang_server_name}")
        async with asyncssh.connect(
            sglang_server_name,
            username=os.environ.get('SSH_USER', os.getlogin()),
            port=22,
            client_keys=[SSH_KEY],
            known_hosts=None,
        ) as conn:
            listener = await conn.forward_local_port('127.0.0.1', 0, '127.0.0.1', get_inference_port(args))
            local_port = listener.get_port()

            # Ensure the httpx/OpenAI client closes BEFORE event loop exits
            # Use a long timeout to prevent client-side cancellations during long inference
            async with openai.AsyncOpenAI(
                api_key='EMPTY',
                base_url=f'http://localhost:{local_port}/v1',
                timeout=float(client_timeout)
            ) as client:
                sem = asyncio.Semaphore(max_concurrency)

                async def run_one(msgs):
                    async with sem:
                        return await fetch_completion(client, msgs, sampling_params)

                tasks = [asyncio.create_task(run_one(msgs)) for msgs in all_msgs]
                results = await asyncio.gather(*tasks, return_exceptions=False)
                return results  # may include Exception objects from last-attempt failures

def inference(all_msgs, sampling_params, batch_size, args):
    print(f"   Total messages: {len(all_msgs)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sampling params: {sampling_params}")

    # Enhanced logging for LoRA path
    if 'extra_body' in sampling_params and 'lora_path' in sampling_params['extra_body']:
        lora_path = sampling_params['extra_body']['lora_path']
        print(f"   🔍 CLIENT SENDING LoRA REQUEST:")
        print(f"      LoRA Path in Request: {lora_path}")
        print(f"      Path exists: {os.path.exists(lora_path)}")
    else:
        print(f"   📝 No LoRA path in request (using base model)")

    # Debug mode: return fake responses without starting server
    if getattr(args, 'debug_mode', False):
        print('🐛 DEBUG MODE: Using stubbed inference responses')
        return debug_inference(all_msgs, sampling_params)

    if args.sglang_server_name is None:
        print(f"   ℹ️  No server running, starting new SGLang server...")
        stop_sglang_server(args, args.sglang_process)
        args.sglang_process, args.sglang_server_name = start_sglang_server(args, args.model, args.inf_n_gpus, lora_model='', log_path=join(args.sglang_log_basepath, f'iter{args.iteration}'))
    else:
        print(f"   ✅ Using existing SGLang server: {args.sglang_server_name}")
    # print(f"   ✅ Using existing SGLang server: {args.sglang_server_name}")

    # Single event loop for the whole job
    async def _run():
        # If you still want batching, do it here INSIDE the async world
        out = []
        num_batches = (len(all_msgs) + batch_size - 1) // batch_size
        print(f"   📦 Processing {num_batches} batches...")
        for i in tqdm(range(0, len(all_msgs), batch_size), desc='Inference batches'):
            batch = all_msgs[i:i+batch_size]
            out.extend(await inference_async(batch, sampling_params, args.sglang_server_name, args=args))
        print(f"   ✅ Completed {len(out)} inference calls")
        return out

    return asyncio.run(_run())


def inference_and_eval(args, ds, iteration, lora_path, is_alphaverus):
    ## Model name
    print('Inference info')
    lora_model = lora_path if lora_path != args.model else ''

    # Convert to absolute path for SGLang LoRA loading (must match server's absolute path)
    if lora_model:
        import os
        lora_path_abs = os.path.abspath(lora_path)
    else:
        lora_path_abs = lora_path

    lora_add = {} if lora_model=='' else {'extra_body': {'lora_path': lora_path_abs}}
    if lora_path != args.model:
        print(f'   LoRA path: {lora_path_abs}')
        assert iteration >= 1, f'Iteration is: {iteration} | Lora path is: {lora_path}'
        model_name = lora_path.replace('models/', '')
    else:
        # This is the base model (iteration 0)
        assert iteration==0 or is_alphaverus
        model_name = generate_model_name_for_cache(args.model, iteration, proposal_strategy=args.proposal_strategy, train_dataset=args.train_dataset, epochs=args.epochs, inf_fs=args.inf_fs, inf_k=args.inf_k)
    print(f'   Model name for cache: {model_name}')
    print(f'   Dataset size: {len(ds)} questions')

    ## Caching - ATOMIC: either all questions cached or regenerate all
    required_q_ids = set(ds.question_id.tolist())

    if getattr(args, 'skip_cache', False):
        print(f"⏭️  Skipping cache read (--skip_cache enabled)")
        cache_results = {}
        uncached_q_ids = list(required_q_ids)
    else:
        # Load cached inferences for this iteration
        inference_cache = args.cache.load_inferences(iteration)
        if inference_cache:
            cache_results = inference_cache.to_dict()
            cached_q_ids = set(cache_results.keys())

            # Check if cache is complete for current dataset
            if cached_q_ids >= required_q_ids:
                # Cache is complete - use it
                print(f"✅ Cache COMPLETE: All {len(required_q_ids)} results found for iteration {iteration}")
                uncached_q_ids = []
            else:
                # Cache is incomplete - regenerate everything atomically
                missing = required_q_ids - cached_q_ids
                print(f"⚠️  Cache INCOMPLETE: {len(cached_q_ids)} cached but {len(required_q_ids)} needed (missing {len(missing)} questions)")
                print(f"    Regenerating all {len(required_q_ids)} questions atomically...")
                uncached_q_ids = list(required_q_ids)
                cache_results = {}  # Clear partial cache
        else:
            print(f"📭 No cached inference results found for iteration {iteration}")
            cache_results = {}
            uncached_q_ids = list(required_q_ids)

    ## Inference for what was uncached
    if len(uncached_q_ids) == 0:
        print(f"✅ Cache HIT: All {len(ds)} inference results found in cache for model '{model_name}'")
        new_cache_results = {}
    else:
        if getattr(args, 'skip_cache', False):
            print(f"🔄 Running fresh inference for all {len(ds)} questions (skip_cache enabled)")
        else:
            print(f"📊 Cache status: {len(cache_results.keys())}/{len(ds)} cached ({len(cache_results.keys())/len(ds)*100:.1f}%), {len(uncached_q_ids)} need inference")

        # Restart SGLang server with new model (skip in debug mode)
        if not getattr(args, 'debug_mode', False):
            print(f'🔧 Restarting SGLang server for model: {args.model}' + (f' + LoRA: {lora_path_abs}' if lora_model else ''))
            # Use absolute path for both server start and client requests
            stop_sglang_server(args, args.sglang_process)
            args.sglang_process, args.sglang_server_name = start_sglang_server(args, args.model, args.inf_n_gpus, lora_model=lora_path_abs if lora_model else '', log_path=join(args.sglang_log_basepath, f'iter{iteration}'))
        else:
            print('🐛 DEBUG MODE: Skipping SGLang server management')
        # print('🔧 Using existing SGLang server (skipping restart)' + (f' with LoRA request: {lora_path_abs}' if lora_model else ''))

        # Run inference only on uncached specs
        uncached_ds = ds.loc[ds.question_id.isin(uncached_q_ids)]
        # uncached_ds = ds.loc[ds.task=='MBPP'] # TODO: change
        uncached_specs = uncached_ds['spec'].tolist()
        uncached_q_ids = uncached_ds['question_id'].tolist()
        if len(uncached_q_ids) > 0:
            # Get spec-completion pairs once for all specs (filtering happens in get_inf_prompt)
            spec_completion_pairs = get_spec_completion_pairs(ds=ds, use_human_data=args.inf_use_human_data)
            print(f"   Generated {len(spec_completion_pairs)} spec-completion pairs for few-shot prompting")

            # Generate k varied prompts per spec (each with different few-shot examples)
            all_msgs = []
            question_idxs = []
            specs = []
            for q_idx, spec in enumerate(tqdm(uncached_specs, desc='Making fs prompts by sampling from ds')):
                for k_idx in range(args.inf_k):
                    # Each call to get_inf_prompt samples different few-shot examples from the pairs # (filtering out the current spec happens inside get_inf_prompt)
                    prompt = get_inf_prompt(spec_in=spec, k=args.inf_fs, spec_completion_pairs=spec_completion_pairs)
                    all_msgs.append(prompt)
                    question_idxs.append(q_idx)
                    specs.append(spec)
            
            # Create list to store postprocessed completions
            postprocessed_completions = []

            # Create pass function that has specs in closure (for spec + completion evaluation)
            pass_fn = make_completion_pass_fn(specs, args.inf_k, postprocessed_out=postprocessed_completions)

            # Run inference and evaluation with varied prompts
            new_cache_results_list = inference_and_passk(
                all_msgs=all_msgs,
                sampling_params={"model": "", "temperature": args.inf_temp, "max_tokens": args.inf_max_tokens, "timeout": args.inf_client_timeout, "seed": args.seed, **lora_add},
                question_idxs=np.array(question_idxs),
                pass_fn=pass_fn,
                args=args,
                inf_bs=args.inf_bs,
                return_format='cache_results',
                postprocessed_responses=postprocessed_completions,  # Store postprocessed versions
            )
            
            # Save new results to cache (only the uncached ones)
            print(f'📊 Saving {len(new_cache_results_list)} new inference results to cache...')

            # Convert dict results to InferenceResult objects
            new_inference_results = []
            for cache_result, question_id in zip(new_cache_results_list, uncached_q_ids):
                # Get additional info from ds
                ds_row = uncached_ds.loc[uncached_ds.question_id == question_id].iloc[0]

                result = InferenceResult(
                    question_id=question_id,
                    question_spec=ds_row['spec'],
                    dataset=ds_row.get('task', ''),
                    nl_desc=ds_row.get('nl_desc', ''),
                    gold_code=ds_row.get('gold_code', ''),
                    model_name=model_name,
                    pass_rate=cache_result['pass_rate'],
                    n_passing=cache_result['n_passing'],
                    n_total=cache_result['n_total'],
                    passing_solution=cache_result.get('passing_solution', ''),
                    failing_solution=cache_result.get('failing_solution', ''),
                    all_solutions=cache_result.get('all_solutions', []),
                    pass_at_k=cache_result.get('pass_at_k', {}),
                    timestamp=__import__('datetime').datetime.now().isoformat(),
                    task_id=ds_row.get('task_id', None),
                    parent=ds_row.get('parent', None),
                    ancestor=ds_row.get('ancestor', None),
                    is_train_dataset=ds_row.get('is_train_dataset', False),
                    is_test_dataset=ds_row.get('is_test_dataset', False),
                )
                new_inference_results.append(result)

            new_cache_results = {k:v for k,v in zip(uncached_q_ids, new_cache_results_list)}

    ## Save inference cache atomically (only for current dataset)
    if len(uncached_q_ids) > 0:
        # ATOMIC SAVE: Only save results for current dataset, not accumulated old results
        # This ensures the cache always reflects exactly what was requested
        all_inference_results = new_inference_results

        # Create and save InferenceCache
        complete_cache = InferenceCache(
            iteration=iteration,
            model_name=model_name,
            results=all_inference_results,
            metadata={
                'inf_k': args.inf_k,
                'inf_fs': args.inf_fs,
                'inf_temp': args.inf_temp,
                'inf_top_p': args.inf_top_p,
                'inf_max_tokens': args.inf_max_tokens,
                'base_model': args.model,
                'lora_path': lora_path if lora_path != args.model else '',
            },
            timestamp=__import__('datetime').datetime.now().isoformat()
        )
        args.cache.save_inferences(complete_cache)
        print(f'✅ Successfully cached {len(all_inference_results)} inference results for iteration {iteration} (atomic save)')

    ## Update ds
    assert set(cache_results.keys()) & set(new_cache_results.keys()) == set()

    # Drop duplicates to prevent assertion failures
    original_len = len(ds)
    ds = ds.drop_duplicates(subset=['question_id'], keep='first')
    if len(ds) < original_len:
        print(f"⚠️  Dropped {original_len - len(ds)} duplicate rows based on question_id")

    # for q_id in uncached_ds.question_id.tolist():
    for q_id in ds.question_id.tolist():
        if not (ds.question_id == q_id).sum() == 1:
            # save ds to temp file
            ds.to_csv(f'./temp/hi.csv', index=False)
            assert False, f'Multiple rows found for question id: {q_id}'
        # Handle both InferenceResult objects and dicts
        if q_id in cache_results:
            cache_obj = cache_results[q_id]
            cache_result = cache_obj.__dict__ if hasattr(cache_obj, '__dict__') else cache_obj
        else:
            cache_result = new_cache_results[q_id]

        idx = ds.index[ds.question_id == q_id][0]
        for k, v in cache_result.items():
            if k == 'pass_at_k':
                for k2, v2 in v.items():
                    ds.at[idx, k2] = v2
            elif k == 'all_solutions':
                v = '||____||'.join(v) if isinstance(v, list) else v
            else:
                ds.at[idx, k] = v
    ds['solved'] = ds.n_passing > 0

    ## Update metrics
    metrics = {}
    # for task in ['MBPP']:
    for task in ds.task.unique().tolist()+['all']:
        sub_ds= ds[ds['task']==task] if task!='all' else ds

        # Calculate pass@k using the correct estimator for each problem
        pass_at_k_metrics = {}
        for k in K_LIST:
            if k > args.inf_k:
                break
            pass_at_k_metrics[f'{task}/pass@{k}'] = estimate_pass_at_k( sub_ds['n_total'].values, sub_ds['n_passing'].values, k ).mean()

        metrics = {
            **metrics,
            **pass_at_k_metrics,
            f'{task}/n_problems': len(sub_ds),
            f'{task}/n_solved': sub_ds['solved'].sum(),
            f'{task}/perc_solved': sub_ds['solved'].mean(),
            # f'{task}/pass_rate': wandb.Histogram(sub_ds['pass_rate'].to_numpy()),
        }

    print(f'\n--- Inference & Eval Metrics ---')
    pprint(metrics)
    print(f'------')

    return ds, metrics