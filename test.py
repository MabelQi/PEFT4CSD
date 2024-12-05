import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    matthews_corrcoef
)

parser = argparse.ArgumentParser("")
parser.add_argument("--model_name", type=str, default="codebert-base")
parser.add_argument("--model_name_or_path", type=str, default="output/codebert-base/prompt_tuning_seed_42/best_model")
parser.add_argument("--test_file", type=str, default="dataset/testset_CC.csv")
parser.add_argument("--peft", type=bool, default=True)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--results_dir", type=str, default="results")
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)

model_name = args.model_name_or_path.split("/")[-3]

content_write = "=" * 50 + "\n"
content_write += "Inference\n"
content_write += f"model: {model_name}\n"
content_write += f"model_name_or_path: {args.model_name_or_path}\n"
content_write += f"test_file: {args.test_file}\n"
content_write += f"max_seq_length: {args.max_seq_length}\n"
content_write += f"batch_size: {args.batch_size}\n"
content_write += f"results_dir: {args.results_dir}\n"
content_write = "=" * 50 + "\n"
print(content_write)

use_cuda = True

# Load datasets
test_df = pd.read_csv(args.test_file)
testset = Dataset.from_pandas(test_df)

# Get the number of labels
label_list = test_df["label"].unique()
num_labels = len(label_list)

# Load model and tokenizer
padding_side = PADDING_SIDE[model_name]

if args.peft:
    config = PeftConfig.from_pretrained(args.model_name_or_path)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=num_labels
    )

    inference_model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side=padding_side)

# Full fine-tuning
if not args.peft:
    inference_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer_path = f"../CLMs/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side)

if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if "deepseek" or "starcoder" in args.model_name_or_path:
    inference_model.config.pad_token_id = tokenizer.pad_token_id
    inference_model.resize_token_embeddings(len(tokenizer))

# Tokenize datasets
if args.max_seq_length > tokenizer.model_max_length:
    print(
        f"The max_seq_length passed ({args.max_seq_length}) is larger than the mixmum length for the "
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
tokenize_max_length = max_seq_length

if "prompt_tuning" or "prefix_tuning" in args.model_name_or_path:
    if "bert" or "unixcoder" in args.model_name_or_path.lower():
        tokenize_max_length = args.max_seq_length - 20

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=tokenize_max_length)
    return outputs

tokenized_datasets = testset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "text_label"],
    load_from_cache_file=False
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    outputs = tokenizer.pad(examples, return_tensors="pt", padding=True, max_length=max_seq_length)
    return outputs

test_dataloader = DataLoader(
    tokenized_datasets,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


# Test
def evaluate(predictions, references):
    predictions = predictions.cpu().numpy()
    references = references.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average='macro', zero_division=1)
    mcc = matthews_corrcoef(references, predictions)
    return [precision, recall, f1, mcc]

if use_cuda:
    inference_model.cuda()

inference_model.eval()

predictions = []
references = []

progress_bar_test = tqdm(
    total=len(test_dataloader),
    desc=f"Test",
    position=0,
    mininterval=1,
    leave=True
)

for step, batch in enumerate(tqdm(test_dataloader)):
    batch = {k: v.cuda() for k, v in batch.items()} if use_cuda else batch
    with torch.no_grad():
        outputs = inference_model(**batch)
        predictions.extend(outputs.logits.argmax(dim=-1))
        references.extend(batch["labels"])

progress_bar_test.close()

predictions = torch.tensor(predictions)
references = torch.tensor(references)
precision, recall, f1, mcc = evaluate(predictions, references)

print(f"Test - Precision: {precision}, Recall: {recall}, F1: {f1}, MCC: {mcc}")

results_data = {
    "Model": model_name,
    "Model path": args.model_name_or_path,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "MCC": mcc
}

temp = args.model_name_or_path.split("/")[-2]

result_path = os.path.join(args.results_dir, model_name)
os.makedirs(result_path, exist_ok=True)
with open(f"{result_path}/{temp}.json", "w") as json_file:
    json.dump(results_data, json_file, indent=4)
