---
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
library_name: peft
model_name: Qwen_Qwen2.5-Coder-3B-Instruct_iter5_proposal_strat_icl_input_ai_output_ui_fsk_12_easythresh_0.8_medthresh_0.2_hardthresh_0.001_ablation_none_humandata_max_n_qs_10000_base_ds_DAFNY2VERUS_epochs_2_humandata_inf_k_10_inf_fs_1
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for Qwen_Qwen2.5-Coder-3B-Instruct_iter5_proposal_strat_icl_input_ai_output_ui_fsk_12_easythresh_0.8_medthresh_0.2_hardthresh_0.001_ablation_none_humandata_max_n_qs_10000_base_ds_DAFNY2VERUS_epochs_2_humandata_inf_k_10_inf_fs_1

This model is a fine-tuned version of [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.15.2
- TRL: 0.19.1
- Transformers: 4.51.1
- Pytorch: 2.6.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```