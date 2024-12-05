from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LORA_IA3_TARGET_MODULES = {
    "codebert-base": {
        "target_modules_lora": ["query", "key", "value"],
        "target_modules_ia3": ["query", "key", "value", "output.dense"],
        "ff_modules": ["output.dense"]
    },
    "graphcodebert-base": {
        "target_modules_lora": ["query", "key", "value"],
        "target_modules_ia3": ["query", "key", "value", "output.dense"],
        "ff_modules": ["output.dense"]
    },
    "unixcoder-base": {
        "target_modules_lora": ["query", "key", "value"],
        "target_modules_ia3": ["query", "key", "value", "output.dense"],
        "ff_modules": ["output.dense"]
    },
    "codet5-base": {
        "target_modules_lora": ["q", "k", "v"],
        "target_modules_ia3": ["q", "k", "v", "wo"],
        "ff_modules": ["wo"]
    },
    "deepseek-coder-1.3b-base": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "down_proj"],
        "ff_modules": ["down_proj"]
    },
    "deepseek-coder-6.7b-base": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "down_proj"],
        "ff_modules": ["down_proj"]
    },
    "starcoderbase-1b": {
        "target_modules_lora": ["c_attn", "c_proj"],
        "target_modules_ia3": ["c_attn", "c_proj", "mlp.c_proj"],
        "ff_modules": ["mlp.c_proj"]
    },
    "starcoderbase-3b": {
        "target_modules_lora": ["c_attn", "c_proj"],
        "target_modules_ia3": ["c_attn", "c_proj", "mlp.c_proj"],
        "ff_modules": ["mlp.c_proj"]
    },
    "starcoderbase-7b": {
        "target_modules_lora": ["c_attn", "c_proj"],
        "target_modules_ia3": ["c_attn", "c_proj", "mlp.c_proj"],
        "ff_modules": ["mlp.c_proj"]
    },
    "codellama-7b-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "down_proj"],
        "ff_modules": ["down_proj"]
    }
}

PADDING_SIDE = {
    "codebert-base": "right",
    "graphcodebert-base": "right",
    "unixcoder-base": "right",
    "codet5-base": "right",
    "codet5-large": "right",
    "deepseek-coder-1.3b-base": "left",
    "deepseek-coder-6.7b-base": "left",
    "starcoderbase-1b": "left",
    "starcoderbase-3b": "left",
    "starcoderbase-7b": "left",
    "codellama-7b-hf": "left"
}

def load_trainset(train_file, max_train_samples=None, seed=42):
    train_df = pd.read_csv(train_file)

    if max_train_samples != None:
        sample_df = pd.DataFrame()
        total_samples = train_df.shape[0]
        label_counts = train_df["label"].value_counts()

        for label, item in label_counts.items():
            sample_count = int(item / total_samples * max_train_samples)
            label_df = train_df[train_df["label"] == label].sample(n=sample_count, replace=False, random_state=seed, ignore_index=True)
            sample_df = pd.concat([sample_df, label_df], ignore_index=True)

        if sample_df.shape[0] < max_train_samples:
            additional_samples = max_train_samples - sample_df.shape[0]
            additional_df = train_df.sample(n=additional_samples, replace=False, random_state=seed, ignore_index=True)
            sample_df = pd.concat([sample_df, additional_df], ignore_index=True)

        print(f"Sampled {sample_df.shape[0]} samples")

        train_df = sample_df

    return train_df
