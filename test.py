# --- Silence & bypass LiteLLM completely ---
import os, sys, logging, types

# 1) Tell any libs that do see LiteLLM to NOT log
os.environ.setdefault("LITELLM_LOGGING", "OFF")
os.environ.setdefault("LITELLM_LOG", "False")
os.environ.setdefault("LITELLM_SDK_LOGGING", "False")

# 2) If LiteLLM is already imported, drop it; if something tries to import it just for logging,
#    install a tiny stub so imports succeed but do nothing.
if "litellm" in sys.modules:
    del sys.modules["litellm"]

stub = types.ModuleType("litellm")
# provide just enough attributes that some loggers expect
stub.__dict__.update({
    "__version__": "0.0-stub",
    "logging": types.SimpleNamespace(Logging=object),
})
sys.modules["litellm"] = stub

# 3) Turn down noisy loggers
for name in ("litellm", "dspy", "httpx", "openai"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
