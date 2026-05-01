import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# CONFIG
EXCEL_PATH = "all_samples.xlsx"
SHEET_NAME = "data"

OUTDIR = "paper_outputs"
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
N_SPLITS = 5

BATCH_SIZE = 32
EPOCHS = 120
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 18

# Preprocess
ALIGN_TO_CENTER = True         # align by |S11| min to center
USE_UNWRAP_PHASE = True        # unwrap phase
USE_ZSCORE = True              # per-sample per-channel zscore

# Strong Augmentation
AUG_ENABLED = True
AUG_SHIFT_MAX = 10
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_PHASE_OFFSET = 0.25
AUG_GAUSS_STD = 0.01
AUG_BAND_DROP_PROB = 0.35
AUG_BAND_DROP_FRAC = (0.06, 0.18)
AUG_BAND_NOISE_PROB = 0.35
AUG_BAND_NOISE_STD = 0.03

# Plots
TSNE_PERPLEXITY = 18
TSNE_RANDOM_STATE = 42
TSNE_MAX_ITER = 1200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed colors for 9 classes
CLASS_COLORS = [
    "#8FB4DC",  # 0
    "#FFDD8E",  # 1
    "#70CDBE",  # 2
    "#AC99D2",  # 3
    "#7AC3DF",  # 4
    "#F5AA61",  # 5
    "#A1E3F9",  # 6
    "#578FCA",  # 7
    "#92EED8",  # 8
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sorted_cols(cols, prefix):
    picked = [c for c in cols if c.startswith(prefix)]
    picked = sorted(picked, key=lambda x: int(x.split("_")[1]))
    return picked


def load_excel(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "label" not in df.columns:
        raise ValueError("Excel must contain a 'label' column.")
    re_cols = _sorted_cols(df.columns, "Re_")
    im_cols = _sorted_cols(df.columns, "Im_")
    if len(re_cols) == 0 or len(im_cols) == 0:
        raise ValueError("Excel must contain Re_* and Im_* columns (e.g., Re_0..Re_400, Im_0..Im_400).")
    if len(re_cols) != len(im_cols):
        raise ValueError("Re_* and Im_* must have the same length.")

    X_re = df[re_cols].to_numpy(np.float32)  # (N, L)
    X_im = df[im_cols].to_numpy(np.float32)  # (N, L)

    labels = df["label"].astype(str).to_numpy()
    uniq = sorted(np.unique(labels).tolist())
    label_map = {k: i for i, k in enumerate(uniq)}
    inv_map = {i: k for k, i in label_map.items()}
    y = np.array([label_map[v] for v in labels], dtype=np.int64)
    return X_re, X_im, y, label_map, inv_map


def mag_phase(re, im):
    mag = np.sqrt(re * re + im * im).astype(np.float32)
    phase = np.arctan2(im, re).astype(np.float32)
    return mag, phase


def unwrap_phase(phase):
    return np.unwrap(phase, axis=1).astype(np.float32)


def align_by_mag_min(mag, arrays):
    # align each sample so argmin(|S11|) goes to center
    N, L = mag.shape
    center = L // 2
    idx = np.argmin(mag, axis=1)
    shifts = center - idx
    out_arrays = []
    for a in arrays:
        out = np.empty_like(a)
        for i in range(N):
            out[i] = np.roll(a[i], shifts[i])
        out_arrays.append(out)
    return out_arrays


def first_diff(x):
    d = np.diff(x, axis=1)
    d = np.concatenate([d, np.zeros((x.shape[0], 1), dtype=np.float32)], axis=1)
    return d.astype(np.float32)


def per_sample_channel_zscore(X, eps=1e-6):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    return (X - mean) / (std + eps)


class StrongAug:
    def __init__(self, L):
        self.L = L

    def __call__(self, x):
        C, L = x.shape
        out = x.copy()

        # 1) frequency shift
        if AUG_SHIFT_MAX and AUG_SHIFT_MAX > 0:
            shift = np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1)
            if shift != 0:
                out = np.roll(out, shift=shift, axis=1)

        # 2) amplitude scaling (mag & dmag)
        if AUG_SCALE_RANGE is not None:
            s = np.random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
            if C >= 1:
                out[0] *= s
            if C >= 3:
                out[2] *= s

        # 3) phase offset
        if AUG_PHASE_OFFSET and AUG_PHASE_OFFSET > 0 and C >= 2:
            p = np.random.uniform(-AUG_PHASE_OFFSET, AUG_PHASE_OFFSET)
            out[1] += p

        # 4) band drop
        if np.random.rand() < AUG_BAND_DROP_PROB:
            frac = np.random.uniform(AUG_BAND_DROP_FRAC[0], AUG_BAND_DROP_FRAC[1])
            w = max(2, int(frac * L))
            st = np.random.randint(0, L - w)
            for c in range(C):
                m = out[c].mean()
                out[c, st:st + w] = m

        # 5) band noise
        if np.random.rand() < AUG_BAND_NOISE_PROB:
            frac = np.random.uniform(AUG_BAND_DROP_FRAC[0], AUG_BAND_DROP_FRAC[1])
            w = max(2, int(frac * L))
            st = np.random.randint(0, L - w)
            noise = np.random.normal(0, AUG_BAND_NOISE_STD, size=(C, w)).astype(np.float32)
            out[:, st:st + w] += noise

        # 6) global gaussian
        if AUG_GAUSS_STD and AUG_GAUSS_STD > 0:
            out += np.random.normal(0, AUG_GAUSS_STD, size=out.shape).astype(np.float32)

        return out


class SpectralDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
        self.aug = StrongAug(L=X.shape[2]) if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment and AUG_ENABLED:
            x = self.aug(x)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


# ResNet1D (embedding-ready)
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, k=7):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return torch.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, n_classes, in_ch=4, emb_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock1D(64, 64, stride=1, k=7),
            ResBlock1D(64, 64, stride=1, k=7),
        )
        self.layer2 = nn.Sequential(
            ResBlock1D(64, 128, stride=2, k=7),
            ResBlock1D(128, 128, stride=1, k=7),
        )
        self.layer3 = nn.Sequential(
            ResBlock1D(128, 256, stride=2, k=5),
            ResBlock1D(256, 256, stride=1, k=5),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.35)
        self.fc_out = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        emb = self.fc_emb(x)
        logits = self.fc_out(self.dropout(emb))
        return logits

    @torch.no_grad()
    def extract_embedding(self, x):
        self.eval()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        emb = self.fc_emb(x)
        return emb


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


def train_one_fold_with_history(X, y, train_idx, val_idx, n_classes):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    dl_tr = DataLoader(SpectralDataset(X_tr, y_tr, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(SpectralDataset(X_va, y_va, augment=False), batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet1D(n_classes=n_classes, in_ch=X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    best_acc = -1
    best_state = None
    bad = 0

    hist_train_loss = []
    hist_val_acc = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb in dl_tr:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)

        train_loss = total_loss / max(1, total_n)
        val_acc, _, _ = eval_model(model, dl_va)

        hist_train_loss.append(train_loss)
        hist_val_acc.append(val_acc)

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    val_acc, y_true, y_pred = eval_model(model, dl_va)

    return model, val_acc, y_true, y_pred, np.array(hist_train_loss), np.array(hist_val_acc)


def make_features(X_re, X_im):
    # Raw -> mag/phase -> optional unwrap -> align -> diff -> stack -> zscore
    mag, phase = mag_phase(X_re, X_im)
    if USE_UNWRAP_PHASE:
        phase = unwrap_phase(phase)
    if ALIGN_TO_CENTER:
        mag, phase = align_by_mag_min(mag, [mag, phase])

    dmag = first_diff(mag)
    dphase = first_diff(phase)

    X = np.stack([mag, phase, dmag, dphase], axis=1).astype(np.float32)  # (N,4,L)
    if USE_ZSCORE:
        X = per_sample_channel_zscore(X)
    return X


def plot_learning_curve_epoch(train_loss_mat, val_acc_mat, out_png):
    """
    train_loss_mat: list of arrays (var length) => pad by nan
    val_acc_mat: list of arrays (var length) => pad by nan
    """
    max_len = max(len(x) for x in train_loss_mat)

    def pad_nan(arr, L):
        out = np.full((L,), np.nan, dtype=np.float32)
        out[:len(arr)] = arr
        return out

    TL = np.stack([pad_nan(a, max_len) for a in train_loss_mat], axis=0)
    VA = np.stack([pad_nan(a, max_len) for a in val_acc_mat], axis=0)

    tl_mean = np.nanmean(TL, axis=0)
    tl_std = np.nanstd(TL, axis=0)
    va_mean = np.nanmean(VA, axis=0)
    va_std = np.nanstd(VA, axis=0)

    epochs = np.arange(1, max_len + 1)

    fig = plt.figure(figsize=(7, 4.2))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(epochs, va_mean * 100)
    ax1.fill_between(epochs, (va_mean - va_std) * 100, (va_mean + va_std) * 100, alpha=0.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)")

    ax2.plot(epochs, tl_mean)
    ax2.fill_between(epochs, tl_mean - tl_std, tl_mean + tl_std, alpha=0.2)
    ax2.set_ylabel("Training Loss")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_fold_bar(fold_accs, out_png):
    fig = plt.figure(figsize=(5.2, 3.6))
    x = np.arange(1, len(fold_accs) + 1)
    plt.bar(x, np.array(fold_accs) * 100)
    plt.ylim(0, 105)
    plt.xlabel("Fold Index")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_confusion(y_true, y_pred, inv_map, out_prefix):
    n_classes = len(inv_map)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[inv_map[i] for i in labels], columns=[inv_map[i] for i in labels])
    cm_df.to_csv(f"{out_prefix}_counts.csv", encoding="utf-8-sig")

    # row-normalized (%)
    cm_row = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100.0
    cm_row_df = pd.DataFrame(cm_row, index=[inv_map[i] for i in labels], columns=[inv_map[i] for i in labels])
    cm_row_df.to_csv(f"{out_prefix}_rowpct.csv", encoding="utf-8-sig")

    # plot row-normalized heatmap-style
    fig = plt.figure(figsize=(6.2, 5.5))
    plt.imshow(cm_row, aspect="auto")
    plt.colorbar(label="Predicted classes (%)")
    plt.xticks(range(n_classes), [inv_map[i] for i in labels], rotation=45, ha="right")
    plt.yticks(range(n_classes), [inv_map[i] for i in labels])
    plt.xlabel("Predicted value")
    plt.ylabel("Actual value")

    for i in range(n_classes):
        for j in range(n_classes):
            v = cm_row[i, j]
            if v > 0:
                plt.text(j, i, f"{v:.2f}" if v < 100 else "100",
                         ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rowpct.png", dpi=300)
    plt.close(fig)


def save_learning_curve_csv(train_loss_hist, val_acc_hist, out_csv):
    max_len = max(max(len(a) for a in train_loss_hist), max(len(a) for a in val_acc_hist))

    def pad_to(a, L):
        a = np.asarray(a, dtype=np.float32)
        if len(a) < L:
            pad = np.full(L - len(a), np.nan, dtype=np.float32)
            a = np.concatenate([a, pad])
        return a

    TL = np.stack([pad_to(a, max_len) for a in train_loss_hist], axis=0)
    VA = np.stack([pad_to(a, max_len) for a in val_acc_hist], axis=0)

    tl_mean = np.nanmean(TL, axis=0)
    tl_std = np.nanstd(TL, axis=0)
    va_mean = np.nanmean(VA, axis=0)
    va_std = np.nanstd(VA, axis=0)

    df = pd.DataFrame({
        "epoch": np.arange(1, max_len + 1),
        "val_acc_mean": va_mean,
        "val_acc_std": va_std,
        "train_loss_mean": tl_mean,
        "train_loss_std": tl_std,
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def save_embedding_csv(features_2d, y, inv_map, out_csv):
    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "label_id": y,
        "label": [inv_map[int(i)] for i in y],
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def save_overall_summary_csv(fold_accs, y_true, y_pred, out_csv):
    overall_acc = accuracy_score(y_true, y_pred)
    df = pd.DataFrame({
        "metric": ["overall_accuracy", "fold_mean_accuracy", "fold_std_accuracy", "n_folds"],
        "value": [
            float(overall_acc),
            float(np.mean(fold_accs)),
            float(np.std(fold_accs)),
            int(len(fold_accs))
        ]
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def tsne_plot(features_2d, y, inv_map, out_png, title):
    fig = plt.figure(figsize=(6.2, 5.2))

    n_classes = len(inv_map)
    for cls in range(n_classes):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=16,
            color=color,
            label=inv_map[int(cls)]
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def pca_plot(features_2d, y, inv_map, out_png, title):
    fig = plt.figure(figsize=(6.2, 5.2))

    n_classes = len(inv_map)
    for cls in range(n_classes):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=16,
            color=color,
            label=inv_map[int(cls)]
        )

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def compute_tsne(X_flat, y):
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        max_iter=TSNE_MAX_ITER,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X_flat)


def compute_pca(X_flat):
    pca = PCA(n_components=2, random_state=SEED)
    emb2 = pca.fit_transform(X_flat)
    return emb2, pca


def save_pca_meta_csv(pca, out_csv):
    df = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def ablation_build_X(X_re, X_im, mode):
    """
    mode:
      - "s11_db"          => use mag only (1ch) but treat as "dB-like" feature
      - "phase"           => phase only (1ch)
      - "dphase"          => dphase only (1ch)
      - "db+phase"        => mag+phase (2ch)
      - "db+dphase"       => mag+dphase (2ch)
      - "phase+dphase"    => phase+dphase (2ch)
      - "all"             => mag+phase+dmag+dphase (4ch)
    """
    mag, phase = mag_phase(X_re, X_im)
    if USE_UNWRAP_PHASE:
        phase = unwrap_phase(phase)
    if ALIGN_TO_CENTER:
        mag, phase = align_by_mag_min(mag, [mag, phase])

    dmag = first_diff(mag)
    dphase = first_diff(phase)

    if mode == "s11_db":
        X = mag[:, None, :]
    elif mode == "phase":
        X = phase[:, None, :]
    elif mode == "dphase":
        X = dphase[:, None, :]
    elif mode == "db+phase":
        X = np.stack([mag, phase], axis=1)
    elif mode == "db+dphase":
        X = np.stack([mag, dphase], axis=1)
    elif mode == "phase+dphase":
        X = np.stack([phase, dphase], axis=1)
    elif mode == "all":
        X = np.stack([mag, phase, dmag, dphase], axis=1)
    else:
        raise ValueError("Unknown ablation mode")

    X = X.astype(np.float32)
    if USE_ZSCORE:
        X = per_sample_channel_zscore(X)
    return X


def run_cv(X, y, n_classes):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_accs = []
    all_true, all_pred = [], []
    train_loss_hist = []
    val_acc_hist = []
    best_models = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model, acc, y_true, y_pred, tl, va_curve = train_one_fold_with_history(X, y, tr, va, n_classes)
        fold_accs.append(acc)
        all_true.append(y_true)
        all_pred.append(y_pred)
        train_loss_hist.append(tl)
        val_acc_hist.append(va_curve)
        best_models.append(model)
        print(f"[Fold {fold}] val_acc = {acc:.4f}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return fold_accs, y_true, y_pred, train_loss_hist, val_acc_hist, best_models


def main():
    set_seed(SEED)

    # Load & preprocess
    X_re, X_im, y, label_map, inv_map = load_excel(EXCEL_PATH, SHEET_NAME)
    n_classes = len(label_map)
    print("Classes:", n_classes, label_map)
    print("Device:", DEVICE)

    # Main model (ALL features)
    X = make_features(X_re, X_im)  # (N,4,L)

    fold_accs, y_true, y_pred, train_loss_hist, val_acc_hist, models = run_cv(X, y, n_classes)

    # Save fold accs
    pd.DataFrame({"fold": np.arange(1, N_SPLITS + 1), "val_acc": fold_accs}).to_csv(
        os.path.join(OUTDIR, "fold_accuracy.csv"), index=False, encoding="utf-8-sig"
    )

    # Save overall summary
    save_overall_summary_csv(
        fold_accs, y_true, y_pred,
        out_csv=os.path.join(OUTDIR, "overall_summary.csv")
    )

    # Confusion matrix outputs
    save_confusion(
        y_true, y_pred, inv_map,
        out_prefix=os.path.join(OUTDIR, "confusion_matrix")
    )

    # Epoch learning curve (val acc + train loss)
    plot_learning_curve_epoch(
        train_loss_hist,
        val_acc_hist,
        out_png=os.path.join(OUTDIR, "learning_curve_epoch_acc_loss.png")
    )

    save_learning_curve_csv(
        train_loss_hist,
        val_acc_hist,
        out_csv=os.path.join(OUTDIR, "learning_curve_epoch_acc_loss.csv")
    )

    # Fold index bar
    plot_fold_bar(
        fold_accs,
        out_png=os.path.join(OUTDIR, "fold_index_accuracy.png")
    )

    # t-SNE BEFORE training
    X_flat_before = X.reshape(X.shape[0], -1)
    emb2_before = compute_tsne(X_flat_before, y)
    tsne_plot(
        emb2_before, y, inv_map,
        out_png=os.path.join(OUTDIR, "tsne_before_training.png"),
        title="t-SNE (Before Training, Preprocessed Features)"
    )
    save_embedding_csv(
        emb2_before, y, inv_map,
        out_csv=os.path.join(OUTDIR, "tsne_before_training_points.csv")
    )

    # PCA BEFORE training
    emb2_pca, pca_model = compute_pca(X_flat_before)
    pca_plot(
        emb2_pca, y, inv_map,
        out_png=os.path.join(OUTDIR, "pca_before_training.png"),
        title="PCA (Preprocessed Features)"
    )
    save_embedding_csv(
        emb2_pca, y, inv_map,
        out_csv=os.path.join(OUTDIR, "pca_before_training_points.csv")
    )
    save_pca_meta_csv(
        pca_model,
        out_csv=os.path.join(OUTDIR, "pca_explained_variance.csv")
    )

    # t-SNE AFTER training
    best_fold = int(np.argmax(fold_accs))
    model_best = models[best_fold].to(DEVICE)
    model_best.eval()

    dl_all = DataLoader(SpectralDataset(X, y, augment=False), batch_size=128, shuffle=False)
    embs = []
    with torch.no_grad():
        for xb, _ in dl_all:
            xb = xb.to(DEVICE)
            e = model_best.extract_embedding(xb).cpu().numpy()
            embs.append(e)
    embs = np.concatenate(embs, axis=0)  # (N, emb_dim)

    emb2_after = compute_tsne(embs, y)
    tsne_plot(
        emb2_after, y, inv_map,
        out_png=os.path.join(OUTDIR, "tsne_after_training.png"),
        title="t-SNE (After Training, ResNet1D Embedding)"
    )
    save_embedding_csv(
        emb2_after, y, inv_map,
        out_csv=os.path.join(OUTDIR, "tsne_after_training_points.csv")
    )

    # Ablation study
    ablation_modes = [
        ("s11_db", "s11_dB"),
        ("phase", "phase"),
        ("dphase", "dphase"),
        ("db+phase", "dB+phase"),
        ("db+dphase", "dB+dphase"),
        ("phase+dphase", "phase+dphase"),
        ("all", "all"),
    ]

    ab_rows = []
    for mode, label in ablation_modes:
        print(f"\n[Ablation] mode = {label}")
        X_ab = ablation_build_X(X_re, X_im, mode)
        fold_accs_ab, _, _, _, _, _ = run_cv(X_ab, y, n_classes)
        ab_rows.append({
            "mode": label,
            "acc_mean": float(np.mean(fold_accs_ab)),
            "acc_std": float(np.std(fold_accs_ab)),
        })

    ab_df = pd.DataFrame(ab_rows)
    ab_df.to_csv(os.path.join(OUTDIR, "ablation_study.csv"), index=False, encoding="utf-8-sig")

    # Ablation bar plot
    fig = plt.figure(figsize=(7.2, 4.0))
    x = np.arange(len(ab_df))
    plt.bar(x, ab_df["acc_mean"] * 100, yerr=ab_df["acc_std"] * 100, capsize=4)
    plt.xticks(x, ab_df["mode"], rotation=30, ha="right")
    plt.ylabel("Average accuracy (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ablation_study.png"), dpi=300)
    plt.close(fig)

    print("\nDone. Outputs saved in:", OUTDIR)


if __name__ == "__main__":
    main()