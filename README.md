# A Comprehensive Evaluation of Parameter-Efficient Fine-Tuning on Code Smell Detection

Replication package for our paper, "A Comprehensive Evaluation of Parameter-Efficient Fine-Tuning on Code Smell Detection", submitted to TOSEM. In this README, we provide comprehensive instructions on setting up the repository and running the experiments presented in our paper. The code is designed to be easily adapted for further exploration of parameter-efficient fine-tuning methods applied to Large Language Models (LLMs) for other classification tasks.

## Directory Structure of the Repo

- `train` folder

  contains 5 python files, including `prompt_tuning.py`, `prefix_tuning.py`, `lora.py`, `IA3.py`, and  `full_fine_tuning.py`. These files implement different PEFT methods as well as the full fine-tuning approach for training language models.

- `utils.py`

  includes utility functions that are commonly used across the training scripts.

- `test.py`

  evaluates the fine-tuned language models, either using PEFT methods or full fine-tuning.

- `dataset` folder

  contains the training, validation, and test sets for four types of code smells at both the method and class level: *Complex Conditional (CC)*, *Complex Method (CM)*, and *Data Class (DC)*. Additionally, it includes training sets with 50, 100, 250, and 500 samples for low-resource scenarios.

- `results` folder

  contains the results of RQ1, RQ2, RQ3, and RQ4. Additionally, the subfolder for RQ4 contains the templates for prompt 2 and prompt 3.

- `requirements.txt`

  contains a list of all the Python packages required for running the experiments in this repo.

​	

## Installation

1. Clone the repo using `git clone` command.

2. Using Conda to create a `Python 3.8` virtual environment and install the dependencies.

   ```
   conda create -n myenv python=3.8
   conda activate myenv
   pip install -r requirements.txt
   ```

   Note that we run all the experiments on a single a 48G NVIDIA RTX 6000 Ada Generation GPU.

   

## Training the models

### Fine tune a LLM using a specific PEFT method

``` 
CUDA_VISIBLE_DEVICES=0 python train/prompt_tuning.py \
	--seed 42 \
  	--model_name_or_path ../CLMs/codebert-base \
  	--train_file dataset/trainset_CC.csv \
  	--valid_file dataset/validset_CC.csv \
  	--max_seq_length 1024 \
	--batch_size 1 \
	--num_epochs 10 \
	--output_dir output/RQ1/CC
```

You can experiment with different PEFT methods, including prompt tuning, prefix tuning, LoRA, and (IA)³, by modifying the Python file accordingly.

### Evaluating a fine-tuned LLM

```
CUDA_VISIBLE_DEVICES=0 python test.py
	--model_name codebert-base
	--model_name_or_path output/codebert-base/prompt_tuning_seed_42/best_model
	--test_file dataset/testset_CC.csv
	--peft True \
	--max_seq_length 1024 \
	--batch_size 4 \
	--results_dir results
```

- `model_name_or_path` specifies the path to the fine-tuned model.
- If the model is fine-tuned with PEFT methods, set `--peft` to `True`. If the model is fully fine-tuned, set `--peft` to `False`.

