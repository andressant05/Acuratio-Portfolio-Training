import os
import json
import time
import difflib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataset_generator import chunk_text, generate_for_chunk, dedupe

def is_duplicate(new_prompt, existing_prompts, threshold=0.9):
    for ep in existing_prompts:
        if difflib.SequenceMatcher(None, new_prompt.lower(), ep.lower()).ratio() > threshold:
            return True
    return False

def process_file(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    text = "\n\n".join(data["chunks"])
    chunks = chunk_text(text, max_chars=3000)
    selected = [c for c in chunks if len(c.strip()) > 300]
    out = []
    seen = set()

    for idx, ch in enumerate(selected):
        qas = generate_for_chunk(ch, questions_per_type=1)
        for qa in qas:
            prompt = qa["prompt"]
            if prompt in seen or is_duplicate(prompt, seen):
                continue
            seen.add(prompt)
            out.append({
                "context": ch.strip(),
                "prompt": prompt,
                "response": qa["response"],
                "metadata": {
                    "source_file": json_path.name,
                    "chunk_index": idx
                }
            })

    return dedupe(out)

def main():
    start = time.time()
    folder = Path("manuela_shower")
    json_files = sorted(folder.glob("*.json"))
    total_files = len(json_files)
    output_dir = Path("temp_outputs")
    output_dir.mkdir(exist_ok=True)
    results = []

    print(f"📁 Found {total_files} files to process.")

    completed = 0
    start_batch = time.time()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_file, jf): jf for jf in json_files}
        for fut in as_completed(futures):
            file = futures[fut]
            try:
                entries = fut.result()
                print(f"✅ {file.name}: {len(entries)} QAs")

                results.extend(entries)

                out_path = output_dir / f"{file.stem}_qas.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for qa in entries:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")

                print(f"💾 Autosaved {len(entries)} QAs from {file.name}")

                completed += 1
                elapsed = time.time() - start_batch
                print(f"📊 Progress: {completed}/{total_files} | ⏱️ {elapsed:.1f}s elapsed ({(completed/total_files)*100:.1f}%)\n")

            except Exception as e:
                print(f"❌ Failed {file.name}: {e}")

    final = dedupe(results)
    print(f"🎯 Total deduped QAs: {len(final)}")

    with open("context_full_dataset.jsonl", "w", encoding="utf-8") as f:
        for qa in final:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"✅ All done – dataset saved to context_full_dataset.jsonl")
    print(f"⏱️ Total time: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
