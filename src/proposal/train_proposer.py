from src.core.server_management import run_sft_training
from os.path import join, exists
from src.core.io_utils import save_jsonl


trained_proposer_prompt = [
    { 'role': 'system', 'content': 'You are a helpful and knowledgeable programming assistant with a strong background in Verus (Rust) and conversant in natural language.' },
    { "role": "user", "content": f"Your goal is to output a thorough description of a challenging programming problem in Verus (Rust), complete with details about the function signature, pre-conditions (requires), post-conditions (ensures), input and output types, and function behavior. You may include a description of helper functions you may need to define the spec of the problem, but it is not required." },
]

def train_proposer(args, iteration, ds):
    """
    Train a proposer model for question generation.

    Args:
        args: Command line arguments
        iteration: Current iteration number
        current_model: Current model path/name

    Returns:
        str: Name/path of the trained proposer model
    """
    # Get nl descs within the band of difficulty that we want to replicate
    ds = ds.loc[~ds.is_test_dataset]
    ds = ds.loc[ (ds.pass_rate<=args.tp_pr_upper) & (ds.pass_rate>=args.tp_pr_lower) ]
    ds = ds.loc[ds.nl_desc_oai != '']
    print(f'🎯 Training proposer on {len(ds)} questions')

    # Create training data
    data = [
        {
            'prompt': trained_proposer_prompt,
            'completion': [{'role': 'assistant', 'content': nl_desc}]
        }
        for nl_desc in ds['nl_desc_oai'].tolist()
    ]
    print(f'Length of SFT data: {len(data)}')
    data_path = join(args.sft_basepath, f'train_proposer_iter_{iteration}_max_n_qs_{args.max_n_qs}.jsonl')
    save_jsonl(data_path, data)

    proposal_model_path = f"models/proposer_iter{iteration}_max_n_qs_{args.max_n_qs}_model_{args.model.replace('/', '_').replace('-', '_')}"
    if exists(proposal_model_path):
        print(f'✅ Proposer model already exists at {proposal_model_path}, loading from there.')
    else:
        print(f'🚀 Training proposer model at {proposal_model_path}')
        log_path = join(args.trn_log_basepath, f'proposer_iter{iteration}')
        run_sft_training(
            model=args.model,
            output_path=proposal_model_path,
            data_path=data_path,
            log_path=log_path,
            epochs=args.epochs,
            timeout=7200  # 2 hour timeout
        )

    return proposal_model_path
