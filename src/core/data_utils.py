"""Data manipulation utilities for AV2."""
import time
from typing import List, Any
import pandas as pd

def flatten(list_in: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    return [elt2 for elt in list_in for elt2 in elt]

def create_seed_ds(args):
    with Timer("Creating seed dataset", level=3):
        import json
        import hashlib
        import os

        def load_questions_from_file(dataset_name):
            """Load questions from JSONL file in data/ directory."""
            file_path = os.path.join('data', f'{dataset_name}.jsonl')
            questions = []
            with open(file_path, 'r') as f:
                for line in f:
                    q = json.loads(line)
                    # Generate question_id from spec + dataset for consistency
                    qid_str = f"{q['spec']}_{dataset_name}"
                    question_id = hashlib.sha256(qid_str.encode()).hexdigest()
                    questions.append({
                        'spec': q['spec'],
                        'task': dataset_name.upper(),
                        'nl_desc_oai': q.get('nl_desc_oai', ''),
                        'task_id': q.get('task_id', question_id),
                        'question_id': question_id,
                        'gold_code': q.get('gold_code', '')
                    })
            return questions

        # Train
        train_questions = load_questions_from_file(args.train_dataset)
        seed_data = []
        for q in train_questions:
            if q['gold_code'] and not q['gold_code'].startswith(q['spec']):
                print(f'Skipping this base question b/c gold code does not start with spec')
                continue
            seed_data.append({
                'spec': q['spec'],
                'task': q['task'],
                'nl_desc_oai': q['nl_desc_oai'],
                'task_id': q['task_id'],
                'question_id': q['question_id'],
                'is_train_dataset': True,
                'is_test_dataset': False,
                'gold_code': q['gold_code']
            })

        # Add test datasets to the main DataFrame as well
        test_datasets = [ds.strip() for ds in args.test_datasets.split(',')]
        for test_dataset in test_datasets:
            test_questions = load_questions_from_file(test_dataset)
            assert test_questions, f"No questions found for {test_dataset}"
            for q in test_questions:
                seed_data.append({
                    'spec': q['spec'],
                    'task': q['task'],
                    'nl_desc_oai': q['nl_desc_oai'],
                    'task_id': q['task_id'],
                    'question_id': q['question_id'],
                    'is_train_dataset': False,
                    'is_test_dataset': True,
                    'gold_code': q.get('gold_code', '')
                })

        ds = pd.DataFrame(seed_data)
        ds['solved'] = False
        ds['parent'] = ''
        ds['ancestor'] = ''
        ds['passing_solution'] = ''

        # Print dataset statistics
        train_count = len(train_questions)
        print(f"{train_count} train questions from {args.train_dataset}")
        for test_dataset in test_datasets:
            test_count = len(load_questions_from_file(test_dataset))
            print(f"{test_count} test questions from {test_dataset}")

        return ds

class Timer:
    """
    Context-manager timer with an optional label.
    Prints the elapsed wall-clock time when the block exits.
    """
    def __init__(self, name: str = "Timer", level: int = 1, symbol: str = "🚀"):
        self.name = name
        self.level = level
        self.symbol = symbol
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def _get_separator(self) -> str:
        """Get the separator based on the level."""
        if self.level == 1:
            return "=" * 20
        elif self.level == 2:
            return "=" * 10
        elif self.level == 3:
            return "-" * 6
        else:
            return "-" * 3  # Default to level 2

    def __enter__(self):
        sep = self._get_separator()
        print(f"\n{sep} {self.symbol} {self.name} {sep}")
        self.start_time = time.time()
        return self  # lets you inspect .elapsed later if you want

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start_time
        sep = self._get_separator()
        print(f"{sep} 🕒 {self.name} took {self._format_elapsed(self.elapsed)} {sep}\n")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        # < 60s → show seconds (no space, "s")
        if seconds < 60:
            if seconds >= 1:
                val = str(int(seconds))  # floor to whole seconds
            else:
                # keep small sub-second values readable (e.g., 0.123s)
                val = f"{seconds:.3f}".rstrip('0').rstrip('.')
            return f"{val}s"

        # < 1 hour → show minutes to 1 decimal (e.g., 1.5 mins)
        if seconds < 3600:
            mins = seconds / 60
            val = f"{mins:.1f}".rstrip('0').rstrip('.')
            return f"{val} mins"

        # ≥ 1 hour → show hours to 2 decimals (e.g., 1.03 hrs)
        hrs = seconds / 3600
        val = f"{hrs:.2f}".rstrip('0').rstrip('.')
        return f"{val} hrs"
