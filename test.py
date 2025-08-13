# %%capture
%pip install -q dspy-ai pandas numpy tqdm scikit-learn

import os, re, random, inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import dspy
from dspy.teleprompt import BootstrapFewShot

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV  = "training_examples_compelling.csv"
OUT_PROMPT = "improved_prompts_compelling.csv"
OUT_PREDS  = "tuned_predictions_compelling.csv"

# Choose a model you have access to.
# Make sure to set your API key in the environment if using a hosted model.
# os.environ["OPENAI_API_KEY"] = "sk-..."   # uncomment and set if needed
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change to your preferred model

random.seed(7); np.random.seed(7)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV).fillna("")
need = {"initial_prompt", "merchant_document_text", "is_compelling", "confidence_score", "evidence_details"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"CSV must include columns: {sorted(need)}; missing: {sorted(missing)}")

# Ensure types for labels
df["is_compelling"] = df["is_compelling"].astype(bool)
df["confidence_score"] = df["confidence_score"].astype(float)

# -----------------------------
# CONFIGURE DSPy (OpenAI example; swap if needed)
# -----------------------------
if os.getenv("OPENAI_API_KEY"):
    lm = dspy.OpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"))
else:
    raise RuntimeError(
        "Please set OPENAI_API_KEY (for OpenAI) or swap in a different dspy LM client here."
    )

dspy.configure(lm=lm, temperature=0.2, max_tokens=600)

# -----------------------------
# SIGNATURE / MODULE (structured outputs)
# -----------------------------
class PhotoEmailEvidenceSignature(dspy.Signature):
    """Does the merchant data provide photographic evidence such as a copy of an ID or email correspondence between the merchant and cardholder?
    If none of these exist, then the evidence is not compelling."""
    merchant_document_text = dspy.InputField(desc="Extracted text from merchant documents")
    is_compelling = dspy.OutputField(desc="Whether photographic or email evidence is found")
    confidence_score = dspy.OutputField(desc="Confidence score between 0 and 1")
    evidence_details = dspy.OutputField(desc="Specific evidence that was found")

class EvidenceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.Predict(PhotoEmailEvidenceSignature)
    def forward(self, merchant_document_text):
        return self.step(merchant_document_text=merchant_document_text)

# -----------------------------
# HELPERS
# -----------------------------
def make_examples(rows: pd.DataFrame):
    exs = []
    for _, r in rows.iterrows():
        ex = dspy.Example(
            merchant_document_text=str(r["merchant_document_text"]),
            is_compelling=str(bool(r["is_compelling"])),
            confidence_score=str(float(r["confidence_score"])),
            evidence_details=str(r["evidence_details"])
        ).with_inputs("merchant_document_text")
        exs.append(ex)
    return exs

def parse_bool(x: str):
    x = str(x).strip().lower()
    return x in {"true", "yes", "1", "y", "t"}

def parse_float(x: str):
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    try:
        return float(m.group(0)) if m else 0.0
    except:
        return 0.0

# Combined metric: boolean accuracy (40%) + confidence proximity (30%) + evidence F1 (30%)
def metric_fn(example, pred):
    gold_bool = parse_bool(example.is_compelling)
    pred_bool = parse_bool(getattr(pred, "is_compelling", ""))
    acc = 1.0 if gold_bool == pred_bool else 0.0

    gold_c = parse_float(example.confidence_score)
    pred_c = max(0.0, min(1.0, parse_float(getattr(pred, "confidence_score", "0"))))
    conf_score = 1.0 - min(1.0, abs(gold_c - pred_c))

    g = set(str(example.evidence_details).lower().split())
    p = set(str(getattr(pred, "evidence_details", "")).lower().split())
    inter = len(g & p); prec = inter / (len(p) + 1e-9); rec = inter / (len(g) + 1e-9)
    f1 = 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec)

    return 0.4*acc + 0.3*conf_score + 0.3*f1

# --- FIX #1: version‑compatible compile wrapper ---
import inspect
def compile_with_compat(teleprompter, student, trainset, devset):
    sig = inspect.signature(teleprompter.compile)
    params = set(sig.parameters.keys())
    if "valset" in params:
        return teleprompter.compile(student=student, trainset=trainset, valset=devset)
    if "devset" in params:
        return teleprompter.compile(student=student, trainset=trainset, devset=devset)
    if "validset" in params:
        return teleprompter.compile(student=student, trainset=trainset, validset=devset)
    return teleprompter.compile(student=student, trainset=trainset)

# --- FIX #2: safe prompt builder (no backslashes inside f‑string expressions) ---
def build_prompt_string(teleprompter_obj, program_obj):
    instr = PhotoEmailEvidenceSignature.__doc__.strip()

    demos = (
        getattr(teleprompter_obj, "best_demonstrations_", None)
        or getattr(teleprompter_obj, "demonstrations_", None)
        or getattr(program_obj, "demonstrations_", None)
        or []
    )

    blocks = []
    for i, ex in enumerate(demos, 1):
        inp = getattr(ex, "merchant_document_text", getattr(ex, "inputs", {}).get("merchant_document_text", ""))
        ic  = getattr(ex, "is_compelling",    getattr(ex, "outputs", {}).get("is_compelling", ""))
        cs  = getattr(ex, "confidence_score", getattr(ex, "outputs", {}).get("confidence_score", ""))
        ed  = getattr(ex, "evidence_details", getattr(ex, "outputs", {}).get("evidence_details", ""))

        blocks.append(
            f"# Example {i}\n"
            f"Merchant document text:\n{inp}\n"
            f"Expected outputs:\n"
            f"is_compelling: {ic}\n"
            f"confidence_score: {cs}\n"
            f"evidence_details: {ed}"
        )

    examples_text = "\n\n".join(blocks) if blocks else "(teleprompter omitted examples)"

    prompt = (
        "### Instruction\n"
        f"{instr}\n\n"
        "### Few‑Shot Examples\n"
        f"{examples_text}\n\n"
        "### Now answer for a new case.\n"
        "Merchant document text:\n"
        "{merchant_document_text}\n\n"
        "Return fields:\n"
        "- is_compelling: boolean (True/False)\n"
        "- confidence_score: number between 0 and 1\n"
        "- evidence_details: short phrase describing the specific evidence found"
    ).strip()

    return prompt

# -----------------------------
# TRAIN / DEV SPLIT
# -----------------------------
train_df, dev_df = train_test_split(df, test_size=max(1, len(df)//5), random_state=7, shuffle=True)
trainset = make_examples(train_df)
devset   = make_examples(dev_df)

# -----------------------------
# TELEPROMPT (optimize few‑shot)
# -----------------------------
teleprompter = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=6)
program = compile_with_compat(teleprompter, EvidenceModule(), trainset, devset)

# -----------------------------
# EXTRACT & SAVE IMPROVED PROMPT
# -----------------------------
improved_prompt = build_prompt_string(teleprompter, program)
pd.DataFrame([{
    "task": "photo_email_evidence",
    "improved_prompt": improved_prompt,
    "n_train": len(train_df),
    "n_dev": len(dev_df)
}]).to_csv(OUT_PROMPT, index=False)

# -----------------------------
# RUN PREDICTIONS FOR ALL INPUTS & SAVE
# -----------------------------
pred_rows = []
for _, r in df.iterrows():
    pred = program(merchant_document_text=str(r["merchant_document_text"]))
    pred_rows.append({
        "merchant_document_text": r["merchant_document_text"],
        "pred.is_compelling": getattr(pred, "is_compelling", ""),
        "pred.confidence_score": getattr(pred, "confidence_score", ""),
        "pred.evidence_details": getattr(pred, "evidence_details", ""),
        "gold.is_compelling": str(bool(r["is_compelling"])),
        "gold.confidence_score": float(r["confidence_score"]),
        "gold.evidence_details": r["evidence_details"]
    })

pd.DataFrame(pred_rows).to_csv(OUT_PREDS, index=False)

print("✅ Done")
print(f"Saved improved prompt -> {OUT_PROMPT}")
print(f"Saved predictions    -> {OUT_PREDS}")

print("\n--- Improved Prompt (preview) ---\n")
preview = improved_prompt[:1800]
print(preview)
if len(improved_prompt) > 1800:
    print("...\n[truncated]")
