"""Run a single seed for both GatedFusion and ConcatMLP, append to results file."""
import json, sys, logging
from pathlib import Path
from scripts import config

logging.basicConfig(level=logging.WARNING)

seed = int(sys.argv[1])
out_file = Path(sys.argv[2])

results = json.loads(out_file.read_text()) if out_file.exists() else {"gated": [], "concat": []}

# ---------- GatedFusion ----------
import scripts.training.fusion_train as train_mod
gated_out = f"models/fusion/gated_fusion/runs/seed{seed}"
config.GATED_FUSION_OUTPUT_DIR = gated_out
train_mod._output_root = lambda cfg=None: config.RESULTS_DIR / gated_out
cfg = config.get_gated_fusion_config(overrides={"seed": seed})
r = train_mod.train(cfg)
results["gated"].append({
    "seed": seed, "macro_f1": r["macro_f1"], "test_acc": r["test_acc"],
    "per_class_f1": r["per_class_f1"], "epochs": r["epochs_trained"]
})
print(f"[GatedFusion  seed={seed}] F1={r['macro_f1']:.4f}  acc={r['test_acc']:.4f}  epochs={r['epochs_trained']}")

# ---------- ConcatMLP ----------
import scripts.training.concat_train as cb
cb.OUTPUT_DIR = (
    config.RESULTS_DIR / "models" / "fusion" / "concat_mlp" / "runs" / f"seed{seed}"
)
config.SEED = seed
rc = cb.train()
results["concat"].append({
    "seed": seed, "macro_f1": rc["macro_f1"], "test_acc": rc["test_acc"],
    "per_class_f1": rc["per_class_f1"], "epochs": rc["epochs_trained"]
})
print(f"[ConcatMLP    seed={seed}] F1={rc['macro_f1']:.4f}  acc={rc['test_acc']:.4f}  epochs={rc['epochs_trained']}")

out_file.write_text(json.dumps(results, indent=2))
print(f"Saved to {out_file}  (gated={len(results['gated'])}, concat={len(results['concat'])})")
