# %%capture
# 1) Ensure we do NOT use litellm anywhere
%pip uninstall -y litellm || true
# 2) Install only what we need
%pip install -q dspy-ai openai pandas numpy tqdm scikit-learn

import os, re, random, inspect, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import dspy

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV  = "training_examples_compelling.csv"
OUT_PROMPT = "improved_prompts_compelling.csv"
OUT_PREDS  = "tuned_predictions_compelling.csv"
OUT_EVAL   = "eval_and_errors_compelling.csv"

# Use an OpenAI model (no litellm). Make sure OPENAI_API_KEY is set.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # choose a model you have access to

random.seed(7); np.random.seed(7)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV).fillna("")
need = {"initial_prompt", "merchant_document_text", "is_compelling", "confidence_score", "evidence_details"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"CSV must include columns: {sorted(need)}; missing: {sorted(missing)}")

df["is_compelling"] = df["is_compelling"].astype(bool)
df["confidence_score"] = df["confidence_score"].astype(float)

# -----------------------------
# CONFIGURE DSPy — OpenAI adapter ONLY (no litellm)
# -----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Please set OPENAI_API_KEY in your environment before running this cell.")

lm = dspy.OpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm, temperature=0.2, max_tokens=700)

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

def token_f1_str(a: str, b: str) -> float:
    A = set(str(a).lower().split())
    B = set(str(b).lower().split())
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    inter = len(A & B)
    prec = inter / (len(B) + 1e-9)
    rec  = inter / (len(A) + 1e-9)
    return 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec)

# Optimizer metric (robust to DSPy passing a trace: accepts *args)
def metric_fn(example, pred, *args):
    gold_bool = parse_bool(example.is_compelling)
    pred_bool = parse_bool(getattr(pred, "is_compelling", ""))
    acc = 1.0 if gold_bool == pred_bool else 0.0

    gold_c = parse_float(example.confidence_score)
    pred_c = max(0.0, min(1.0, parse_float(getattr(pred, "confidence_score", "0"))))
    conf_score = 1.0 - min(1.0, abs(gold_c - pred_c))

    f1 = token_f1_str(example.evidence_details, getattr(pred, "evidence_details", ""))

    return 0.4*acc + 0.3*conf_score + 0.3*f1

# Version‑compatible compile wrapper (MIPROv2 signature varies across versions)
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

# Build a clean prompt string (no backslashes inside f‑string expressions)
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
    examples_text = "\n\n".join(blocks) if blocks else "(optimizer omitted examples)"
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
# MIPROv2 — Automatic Instruction Optimizer (ONLY)
# -----------------------------
from dspy.teleprompt import MIPROv2
mipro = MIPROv2(metric=metric_fn, max_iters=3, num_candidates=8, random_seed=7)

# Compile with compatibility wrapper
program = compile_with_compat(mipro, EvidenceModule(), trainset, devset)

# -----------------------------
# EXTRACT & SAVE IMPROVED PROMPT
# -----------------------------
improved_prompt = build_prompt_string(mipro, program)
pd.DataFrame([{
    "optimizer": "MIPROv2 (OpenAI adapter, no litellm)",
    "task": "photo_email_evidence",
    "improved_prompt": improved_prompt,
    "n_train": len(train_df),
    "n_dev": len(dev_df)
}]).to_csv(OUT_PROMPT, index=False)

# -----------------------------
# RUN PREDICTIONS & EVALUATE
# -----------------------------
pred_rows = []
y_true, y_pred = [], []
conf_true, conf_pred = [], []
evi_f1_list = []

for _, r in df.iterrows():
    pred = program(merchant_document_text=str(r["merchant_document_text"]))
    p_bool = parse_bool(getattr(pred, "is_compelling", ""))
    p_conf = max(0.0, min(1.0, parse_float(getattr(pred, "confidence_score", "0"))))
    p_evi  = getattr(pred, "evidence_details", "")

    pred_rows.append({
        "merchant_document_text": r["merchant_document_text"],
        "pred.is_compelling": p_bool,
        "pred.confidence_score": p_conf,
        "pred.evidence_details": p_evi,
        "gold.is_compelling": bool(r["is_compelling"]),
        "gold.confidence_score": float(r["confidence_score"]),
        "gold.evidence_details": r["evidence_details"],
        "confidence_gap": abs(float(r["confidence_score"]) - p_conf),
        "evidence_f1": token_f1_str(r["evidence_details"], p_evi)
    })

    y_true.append(bool(r["is_compelling"]))
    y_pred.append(p_bool)
    conf_true.append(float(r["confidence_score"]))
    conf_pred.append(p_conf)
    evi_f1_list.append(token_f1_str(r["evidence_details"], p_evi))

pd.DataFrame(pred_rows).to_csv(OUT_PREDS, index=False)

prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
mae_conf = float(np.mean(np.abs(np.array(conf_true) - np.array(conf_pred)))) if conf_true else 0.0
avg_evi_f1 = float(np.mean(evi_f1_list)) if evi_f1_list else 0.0

summary_row = pd.DataFrame([{
    "SUMMARY": True,
    "optimizer": "MIPROv2",
    "n_examples": len(df),
    "precision_is_compelling": round(prec, 4),
    "recall_is_compelling": round(rec, 4),
    "f1_is_compelling": round(f1, 4),
    "mae_confidence": round(mae_conf, 4),
    "avg_evidence_f1": round(avg_evi_f1, 4)
}])

errors_df = pd.DataFrame(pred_rows).sort_values(["confidence_gap", "evidence_f1"], ascending=[False, True])
eval_df = pd.concat([summary_row, errors_df], ignore_index=True)
eval_df.to_csv(OUT_EVAL, index=False)

print("✅ Done (MIPROv2, no litellm)")
print(f"Saved improved prompt -> {OUT_PROMPT}")
print(f"Saved predictions    -> {OUT_PREDS}")
print(f"Saved eval+errors    -> {OUT_EVAL}")

print("\n--- Improved Prompt (preview) ---\n")
preview = improved_prompt[:1800]
print(preview)
if len(improved_prompt) > 1800:
    print("...\n[truncated]")

print("\n--- Metrics ---")
print(json.dumps({
    "optimizer": "MIPROv2",
    "precision_is_compelling": round(prec, 4),
    "recall_is_compelling": round(rec, 4),
    "f1_is_compelling": round(f1, 4),
    "mae_confidence": round(mae_conf, 4),
    "avg_evidence_f1": round(avg_evi_f1, 4)
}, indent=2))
