"""
Run CrossAttentionFusion over 5 seeds and print comparison table.
Usage:  python -m scripts.main run cross-attention
"""
import json, random, time
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.models.fusion.cross_attention import CrossAttentionFusion
from scripts.data.fusion_dataset import load_feature_tensors

SEEDS   = [0, 1, 2, 3, 42]
LR      = config.FUSION_LR          # 5e-4
BATCH   = config.FUSION_BATCH_SIZE  # 32
EPOCHS  = config.FUSION_EPOCHS      # 20
PATIENCE = config.FUSION_EARLY_STOPPING_PATIENCE
LABEL_SMOOTHING = config.FUSION_LABEL_SMOOTHING
OUT_DIR = config.RESULTS_DIR / "models" / "fusion" / "cross_attention"
ARTIFACT_DIR = config.ARTIFACTS_DIR / "models" / "fusion" / "cross_attention"
CLASS_NAMES = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_split(split):
    sem, aff, hc, _ = load_feature_tensors(input_config="fused", split=split)
    import pandas as pd
    path = {"train": config.TRAIN_PATH, "val": config.VAL_PATH, "test": config.TEST_PATH}[split]
    labels = torch.tensor(pd.read_csv(path)[config.LABEL_COL].values, dtype=torch.long)
    return sem, aff, hc, labels

def scale_hc(scaler, t, fit=False):
    arr = t.numpy()
    return torch.from_numpy((scaler.fit_transform(arr) if fit else scaler.transform(arr)).astype(np.float32))

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def run_epoch(model, loader, criterion, opt, device):
    training = opt is not None
    model.train(training)
    tot_loss = tot_acc = n = 0
    with torch.set_grad_enabled(training):
        for sem, aff, hc, lbl in loader:
            sem, aff, hc, lbl = sem.to(device), aff.to(device), hc.to(device), lbl.to(device)
            logits = model(sem, aff, hc)
            loss = criterion(logits, lbl)
            if training:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss += loss.item(); tot_acc += accuracy(logits.detach(), lbl); n += 1
    return tot_loss / n, tot_acc / n

def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for sem, aff, hc, lbl in loader:
            logits = model(sem.to(device), aff.to(device), hc.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(preds), np.concatenate(labels)

def collect_attn(model, loader, device):
    """Collect mean attention weights per class (analogous to gate weights)."""
    model.eval()
    all_w, all_lbl = [], []
    with torch.no_grad():
        for sem, aff, hc, lbl in loader:
            _, w = model(sem.to(device), aff.to(device), hc.to(device), return_attn=True)
            all_w.append(w.cpu()); all_lbl.append(lbl)
    w_np = torch.cat(all_w).numpy()    # (N, 2)  [aff, hc]
    l_np = torch.cat(all_lbl).numpy()
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = l_np == i
        if mask.sum() > 0:
            per_class[cls] = w_np[mask].mean(0).tolist()   # [aff_w, hc_w]
    return per_class

def train_one_seed(seed):
    set_seed(seed)
    device = torch.device("cpu")

    sem_tr, aff_tr, hc_tr, lbl_tr = load_split("train")
    sem_va, aff_va, hc_va, lbl_va = load_split("val")
    sem_te, aff_te, hc_te, lbl_te = load_split("test")

    sc = StandardScaler()
    hc_tr = scale_hc(sc, hc_tr, fit=True)
    hc_va = scale_hc(sc, hc_va)
    hc_te = scale_hc(sc, hc_te)

    tr_loader = DataLoader(TensorDataset(sem_tr,aff_tr,hc_tr,lbl_tr), BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(sem_va,aff_va,hc_va,lbl_va), BATCH)
    te_loader = DataLoader(TensorDataset(sem_te,aff_te,hc_te,lbl_te), BATCH)

    model = CrossAttentionFusion().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1; best_state = None; best_epoch = 0
    best_val_loss = float("inf"); no_improve = 0
    history = []

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, opt, device)
        va_loss, va_acc = run_epoch(model, va_loader, criterion, None, device)
        history.append(dict(epoch=ep, tr_loss=round(tr_loss,4), va_loss=round(va_loss,4),
                            tr_acc=round(tr_acc,4), va_acc=round(va_acc,4)))
        if va_acc > best_val_acc:
            best_val_acc = va_acc; best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if va_loss < best_val_loss:
            best_val_loss = va_loss; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    ckpt_dir = ARTIFACT_DIR / "runs" / f"seed{seed}" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    scaler_path = ckpt_dir / "handcrafted_scaler.joblib"
    torch.save(best_state, ckpt_path)
    joblib.dump(sc, scaler_path)

    preds, true = predict(model, te_loader, device)
    macro_f1 = float(f1_score(true, preds, average="macro", zero_division=0))
    test_acc  = float((preds == true).mean())
    per_class_f1 = {CLASS_NAMES[i]: round(float(f1_score(true==i, preds==i, zero_division=0)), 6)
                    for i in range(config.NUM_LABELS)}
    attn_per_class = collect_attn(model, te_loader, device)

    elapsed = time.time() - t0
    print(f"  seed={seed}  F1={macro_f1:.4f}  acc={test_acc:.4f}  "
          f"epochs={len(history)}(best={best_epoch})  {elapsed:.0f}s  params={n_params:,}")

    return {
        "seed": seed, "macro_f1": round(macro_f1,6), "test_acc": round(test_acc,6),
        "per_class_f1": per_class_f1, "epochs": len(history), "best_epoch": best_epoch,
        "n_params": n_params, "attn_per_class": attn_per_class, "history": history,
        "checkpoint_path": str(ckpt_path), "scaler_path": str(scaler_path),
    }

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("CrossAttentionFusion — 5-seed experiment")
    print(f"LR={LR}  batch={BATCH}  patience={PATIENCE}  label_smoothing={LABEL_SMOOTHING}")
    print()

    results = []
    for seed in SEEDS:
        r = train_one_seed(seed)
        results.append(r)

    f1s  = [r["macro_f1"]  for r in results]
    accs = [r["test_acc"]  for r in results]

    print()
    print("=" * 60)
    print("CrossAttentionFusion results (5 seeds):")
    print(f"  Macro F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print()
    print("Per-class F1 means:")
    for cls in CLASS_NAMES:
        vals = [r["per_class_f1"][cls] for r in results]
        print(f"  {cls:<12} {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print()
    print("Comparison:")
    print(f"  MentalRoBERTa (baseline):  F1=0.8873  acc=0.8871  (single run)")
    print(f"  ConcatMLP    (5 seeds):    F1=0.8874±0.0023  acc=0.8886±0.0022")
    print(f"  GatedFusion  (5 seeds):    F1=0.8858±0.0016  acc=0.8870±0.0017")
    print(f"  CrossAttn    (5 seeds):    F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  acc={np.mean(accs):.4f}±{np.std(accs):.4f}")
    print()
    print("Attention weights per class (seed average, [aff_w, hc_w]):")
    for cls in CLASS_NAMES:
        ws = [r["attn_per_class"].get(cls, [0,0]) for r in results]
        mean_aff = np.mean([w[0] for w in ws])
        mean_hc  = np.mean([w[1] for w in ws])
        print(f"  {cls:<12} aff={mean_aff:.4f}  hc={mean_hc:.4f}")

    # Save
    out = {"results": results, "summary": {
        "macro_f1": {"mean": round(np.mean(f1s),4), "std": round(np.std(f1s),4)},
        "test_acc": {"mean": round(np.mean(accs),4), "std": round(np.std(accs),4)},
        "per_class_f1": {
            cls: {"mean": round(np.mean([r["per_class_f1"][cls] for r in results]),4),
                  "std":  round(np.std ([r["per_class_f1"][cls] for r in results]),4)}
            for cls in CLASS_NAMES
        }
    }}
    best_run = max(results, key=lambda r: r["macro_f1"])
    top_ckpt_dir = ARTIFACT_DIR / "checkpoints"
    top_ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_run["checkpoint_path"], top_ckpt_dir / "best.pt")
    shutil.copy2(best_run["scaler_path"], top_ckpt_dir / "handcrafted_scaler.joblib")
    out["summary"]["best_seed"] = best_run["seed"]
    out["summary"]["best_checkpoint_path"] = str(top_ckpt_dir / "best.pt")
    out["summary"]["best_scaler_path"] = str(top_ckpt_dir / "handcrafted_scaler.joblib")
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT_DIR}/summary.json")
