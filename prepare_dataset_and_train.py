import json
from datasets import load_dataset
import nine_to_five  # This assumes your training script is named nine_to_five.py and is in the same folder

# Step 1: Convert dataset into ChatML-like format
input_path = "context_full_dataset.jsonl"
output_path = "full_contexted_manual_trained_dataset.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            item = json.loads(line)
            context = item["context"]
            prompt = item["prompt"]
            response = item["response"]
            formatted = {
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"❌ Skipped a line due to error: {e}")

print("✅ Formatted dataset saved to full_contexted_manual_trained_dataset.jsonl")

# Step 2: Load formatted dataset and train the model
raw_dataset = load_dataset("json", data_files="full_contexted_manual_trained_dataset.jsonl", split="train")

nine_to_five.run_train_on_model(
    hu_fa_data=raw_dataset,
    output="./thursday_temp",
    new_model_name="Llama-3.2-3B-full_contexted_manual_trained_dataset",
    epochs=1,
    batch_size=4,
    resume_from_checkpoint=False  # 🔧 Important fix here!
)
