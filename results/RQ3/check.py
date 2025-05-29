import json
import os

for root, dirs, files in os.walk("RQ3"):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file), 'r') as f:
                data = json.load(f)
                # 改写 "Model path"，从"/home/liangpeng/project/Mabel/PEFTonCodeSmellDetection/output/1st/RQ3/CC/50/deepseek-coder-1.3b-base/full_fine_tuning_seed_42/best_model"只保留“RQ3/CC/50/deepseek-coder-1.3b-base/full_fine_tuning_seed_42/best_model”

                model_path = data["Model path"]
                new_model_path = model_path.split("1st/")[-1]
                data["Model path"] = new_model_path

            with open(os.path.join(root, file), 'w') as f:
                json.dump(data, f, indent=4)    

            print(f"Updated {file} in {root}")
    
print("All JSON files have been updated successfully.")

