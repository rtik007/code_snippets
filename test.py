import os
import warnings
import csv

# List all the models you want to load
model_names = [
    "llama-3.2-1B",
    "llama-3.2-7B",
    "llama-3.2-13B",
    # …add more as needed
]

# Containers to hold successfully loaded models & tokenizers
models = {}
tokenizers = {}

# Prepare CSV logging
log_file = "model_load_log.csv"
fieldnames = ["model_name", "error", "warnings"]
with open(log_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for name in model_names:
        # prepare per‐model directory
        model_dir = os.path.join("models", name)
        os.makedirs(model_dir, exist_ok=True)

        print(f"Loading {name}…")
        warnings_list = []
        error_msg = ""

        # capture all warnings in this block
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            try:
                model, tokenizer = load_model_from_s3(
                    bucket=bucket_name,
                    path_to_model=name,
                    endpoint_url=endpoint,
                    verify_ssl=False,
                    local_path=model_dir,
                    enable_debug=True
                )
                # store only if succeeded
                models[name] = model
                tokenizers[name] = tokenizer
            except Exception as e:
                # record the exception string
                error_msg = str(e)

            # pull out any warnings that were emitted
            for w in caught_warnings:
                warnings_list.append(str(w.message))

        # write one row per model
        writer.writerow({
            "model_name": name,
            "error": error_msg,
            "warnings": "; ".join(warnings_list)
        })

print(f"Done loading.  Log written to {log_file}")

