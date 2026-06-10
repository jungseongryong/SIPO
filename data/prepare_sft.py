import json
import os

import pandas as pd

path = os.environ.get("LUFFY_DATA_DIR", "./")
input_path = os.environ.get("LUFFY_SFT_INPUT", os.path.join(path, "openr1.parquet"))
output_path = os.environ.get("LUFFY_SFT_OUTPUT_DIR", os.path.join(path, "openrlhf_sft"))
output_file = os.path.join(output_path, "train.jsonl")

train_df = pd.read_parquet(input_path)
os.makedirs(output_path, exist_ok=True)

with open(output_file, "w") as f:
    for i in range(len(train_df)):
        if train_df.iloc[i]["target"] is not None:
            train_df.iloc[i]["target"][0]["content"] = train_df.iloc[i]["target"][0]["content"][len("<think>\n"):]
            item = {
                "prompt": train_df.iloc[i]["prompt"].tolist(),
                "target": train_df.iloc[i]["target"].tolist(),
            }
            f.write(json.dumps(item))
            f.write("\n")

print(f"Wrote {output_file}")
