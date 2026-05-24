---
license: mit
base_model: openai-community/gpt2
tags:
- generated_from_trainer
model-index:
- name: financeLM_outputpath_stock_movement_prediction__5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# financeLM_outputpath_stock_movement_prediction__5

This model is a fine-tuned version of [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9480

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 1.1739        | 1.0   | 1817 | 0.9646          |
| 0.8999        | 2.0   | 3634 | 0.9382          |
| 0.8036        | 3.0   | 5451 | 0.9450          |
| 0.7465        | 4.0   | 7269 | 0.9450          |
| 0.7163        | 5.0   | 9085 | 0.9480          |


### Framework versions

- Transformers 4.35.0
- Pytorch 2.1.2+cu121
- Datasets 2.14.5
- Tokenizers 0.14.1
