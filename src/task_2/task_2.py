import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.pop("TF_GPU_ALLOCATOR", None)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras import mixed_precision as mp

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

mp.set_global_policy("mixed_float16")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("./data/A1_data_150") # expects images.npy and labels.npy
ARTIFACTS_DIR = Path("./artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
MAX_EPOCHS = 30

INITIALIZER = "he_uniform"
DROPOUT = 0.3

EPOCH_CSV_PATH = Path("results/epoch_metrics.csv")
TOL_THRESHOLDS = (1, 5, 10)

epoch_stats = {}
results_summary = []

WIDE_BASE = ["model", "epoch", "time_s", "lr"]

KERAS_COLS = [
    "loss", "val_loss",
    "accuracy", "val_accuracy",
    "mae", "val_mae",

    "time_float_mae", "val_time_float_mae",
    "time_angle_vec_mae", "val_time_angle_vec_mae",

    "hour_head_accuracy", "val_hour_head_accuracy",
    "min_head_mae", "val_min_head_mae",
    "hour_head_loss", "val_hour_head_loss",
    "min_head_loss", "val_min_head_loss",
]

CSE_COLS = [
    "train_cse_mean", "train_cse_median", "train_cse_p90",
    "train_acc_le_1m", "train_acc_le_5m", "train_acc_le_10m",
    "val_cse_mean", "val_cse_median", "val_cse_p90",
    "val_acc_le_1m", "val_acc_le_5m", "val_acc_le_10m",
]

ALL_COLS = WIDE_BASE + KERAS_COLS + CSE_COLS

# -------------------- Utilities --------------------
def clear_tf_session():
    tf.keras.backend.clear_session()
    gc.collect()

def DenseK(*a, **k):
    if "kernel_initializer" not in k:
        k["kernel_initializer"] = INITIALIZER
    return L.Dense(*a, **k)

def load_clock_data(data_dir):
    X = np.load(data_dir / "images.npy").astype(np.float32) / 255.0
    y = np.load(data_dir / "labels.npy").astype(np.int32)
    return X, y

def split_indices(no_of_examples, train=0.8, val=0.1, seed=SEED):
    idx = np.arange(no_of_examples)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(no_of_examples * train)
    n_val = int(no_of_examples * val)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

def minutes_after_midnight(hours, minutes):
    return np.asarray(hours) * 60 + np.asarray(minutes)

def circular_minute_distance(a_min, b_min):
    diff = abs(a_min - b_min)
    return min(diff, 720.0 - diff)

def decode_circle(sin_val, cos_val):
    theta = np.arctan2(sin_val, cos_val)
    if theta < 0:
        theta += 2.0 * np.pi
    mins_total = theta / (2.0 * np.pi) * 720.0
    return mins_total % 720.0

def decode_regression(pred_scalar):
    h_float = float(pred_scalar) % 12.0
    total_min = (h_float * 60.0) % 720.0
    return total_min

# -------------------- CSE evaluators --------------------
def eval_common_sense_error_classifier(model, X_sub, y_hm_sub, bin_centers):
    probs = model.predict(X_sub[..., None], verbose=0)
    pred_cls = np.argmax(probs, axis=1)
    pred_total_min = bin_centers[pred_cls].astype(np.float32)
    true_total_min = (minutes_after_midnight(y_hm_sub[:, 0], y_hm_sub[:, 1]) % 720).astype(np.float32)
    cse = np.array([circular_minute_distance(t, p) for t, p in zip(true_total_min, pred_total_min)])
    return cse, pred_total_min

def eval_common_sense_error_regressor(model, X_sub, y_hm_sub):
    preds = model.predict(X_sub[..., None], verbose=0).squeeze()
    pred_total_min = np.array([decode_regression(p) for p in preds], dtype=np.float32)
    true_total_min = (minutes_after_midnight(y_hm_sub[:, 0], y_hm_sub[:, 1]) % 720).astype(np.float32)
    cse = np.array([circular_minute_distance(t, p) for t, p in zip(true_total_min, pred_total_min)])
    return cse, pred_total_min

def eval_common_sense_error_circle(model, X_sub, y_hm_sub):
    vecs = model.predict(X_sub[..., None], verbose=0).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = np.where(norms > 0, vecs / norms, vecs)
    pred_total_min = []
    for (s, c) in vecs:
        pred_total_min.append(decode_circle(s, c))
    pred_total_min = np.array(pred_total_min, dtype=np.float32)
    true_total_min = (minutes_after_midnight(y_hm_sub[:, 0], y_hm_sub[:, 1]) % 720).astype(np.float32)
    cse = np.array([circular_minute_distance(t, p) for t, p in zip(true_total_min, pred_total_min)])
    return cse, pred_total_min

def eval_common_sense_error_multi_circle(model, X, y_hm):
    preds = model.predict(X[..., None], verbose=0)
    y_hour = np.argmax(preds["hour_head"], axis=1)

    v = preds["min_head"].astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(norms > 0, v / norms, v)

    ang = np.arctan2(v[:, 0], v[:, 1])
    ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)
    min_pred = (ang / (2.0 * np.pi) * 60.0)

    total_pred = (y_hour.astype(np.float32) * 60.0 + min_pred) % 720.0
    true_min = (minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32) % 720.0)

    diff = np.abs(total_pred - true_min)
    return np.minimum(diff, 720.0 - diff)

def eval_common_sense_error_multi_regular(model, X, y_hm):
    preds = model.predict(X[..., None], verbose=0)
    y_hour = np.argmax(preds["hour_head"], axis=1)

    min_pred = preds["min_head"].reshape(-1).astype(np.float32)
    min_pred = np.mod(min_pred, 60.0)

    total_pred = (y_hour.astype(np.float32) * 60.0 + min_pred) % 720.0
    true_min = (minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32) % 720.0)

    diff = np.abs(total_pred - true_min)
    return np.minimum(diff, 720.0 - diff)

def _cse_circle(model, X, y):
    cse, _ = eval_common_sense_error_circle(model, X, y)
    return cse

def _cse_reg(model, X, y):
    cse, _ = eval_common_sense_error_regressor(model, X, y)
    return cse

def _cse_clf_bins(model, X, y, bin_centers):
    y_pred = model.predict(X[..., None], verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)
    true_min = (minutes_after_midnight(y[:, 0], y[:, 1]).astype(np.float32) % 720.0)  # wrap to 12h
    pred_min = bin_centers[y_pred_cls].astype(np.float32)
    diff = np.abs(pred_min - true_min)
    return np.minimum(diff, 720.0 - diff)

# -------------------- Datasets --------------------
def minutes_to_unit_vec_60(mins):
    theta = (mins % 60) / 60.0 * (2.0 * np.pi)
    return np.stack([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)

def make_ds_classification(X, y_bins, shuffle=False):
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, y_bins.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_regression_targets(y_hm):
    y_reg = (y_hm[:, 0] % 12).astype(np.float32) + (y_hm[:, 1].astype(np.float32) / 60.0)
    return y_reg

def make_ds_regression(X, y_reg, shuffle=False):
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, y_reg.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_ds_multi(X, y_hm, shuffle=False):
    y_hours = (y_hm[:, 0] % 12).astype(np.int32)
    y_minutes = y_hm[:, 1].astype(np.float32)
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, {"hour_head": y_hours, "min_head": y_minutes}))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_ds_circle(X, y_hm, shuffle=False):
    y_circle = angle_targets_from_labels(y_hm)
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, y_circle.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_ds_multi_circle(X, y_hm, shuffle=False):
    # ensure contiguous and typed arrays on CPU first
    Xc = np.ascontiguousarray(X[..., None].astype(np.float32))
    hrs = np.ascontiguousarray((y_hm[:, 0] % 12).astype(np.int32))
    mins = np.ascontiguousarray(y_hm[:, 1].astype(np.int32))
    y_min_vec = np.ascontiguousarray(minutes_to_unit_vec_60(mins).astype(np.float32))

    # explicitly convert to TF tensors on CPU
    with tf.device("/CPU:0"):
        Xc_tf = tf.convert_to_tensor(Xc, dtype=tf.float32)
        hrs_tf = tf.convert_to_tensor(hrs, dtype=tf.int32)
        minvec_tf = tf.convert_to_tensor(y_min_vec, dtype=tf.float32)

    y = {"hour_head": hrs_tf, "min_head": minvec_tf}
    ds = tf.data.Dataset.from_tensor_slices((Xc_tf, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=0)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------- Models --------------------
def get_backbone_light(input_shape=(75, 75, 1), dropout=DROPOUT):
    inp = Input(input_shape)
    x = L.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(inp)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    return Model(inp, x, name="backbone_light")

def get_backbone_medium(input_shape=(75, 75, 1), dropout=DROPOUT):
    inp = Input(input_shape)
    x = L.Conv2D(64, 3, padding="same", activation="relu", name="conv_b1_1", kernel_initializer=INITIALIZER)(inp)
    x = L.BatchNormalization(name="bn_b1_1")(x)
    x = L.MaxPooling2D(name="pool_b1")(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu", name="conv_b2_1", kernel_initializer=INITIALIZER)(x)
    x = L.BatchNormalization(name="bn_b2_1")(x)
    x = L.MaxPooling2D(name="pool_b2")(x)
    x = L.Conv2D(256, 3, padding="same", activation="relu", name="conv_b3_1", kernel_initializer=INITIALIZER)(x)
    x = L.BatchNormalization(name="bn_b3_1")(x)
    x = L.MaxPooling2D(name="pool_b3")(x)
    x = L.Conv2D(512, 3, padding="same", activation="relu", name="conv_b4_1", kernel_initializer=INITIALIZER)(x)
    x = L.BatchNormalization(name="bn_b4_1")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="drop_gap")(x)
    return Model(inp, x, name="backbone_medium")

def get_backbone_heavy(input_shape=(75, 75, 1), dropout=DROPOUT):
    inp = Input(input_shape)
    x = L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(inp)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(512, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(512, 3, padding="same", activation="relu", kernel_initializer=INITIALIZER)(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    return Model(inp, x, name="backbone_heavy")

def build_head_classification(backbone, num_classes, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    x = DenseK(512, activation="relu")(z)
    x = L.Dropout(DROPOUT)(x)
    logits = DenseK(num_classes)(x)
    out = L.Activation("softmax", name="time_class")(logits)
    model = Model(inp, out, name=f"classifier_{num_classes}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_head_regression(backbone, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    x = DenseK(256, activation="relu")(z)
    x = L.Dropout(DROPOUT)(x)
    out = DenseK(1, name="time_float", dtype="float32")(x)  # fp32 head for mixed precision
    model = Model(inp, out, name="regressor_time")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_head_multi(backbone, lr=1e-3, minute_loss_weight=1.0):
    inp = backbone.input
    z = backbone.output
    shared = DenseK(256, activation="relu", name="mh_shared_dense")(z)
    shared = L.Dropout(DROPOUT, name="mh_shared_do")(shared)
    hour_logits = DenseK(12, name="hour_logits")(shared)
    hour_out = L.Activation("softmax", name="hour_head")(hour_logits)
    min_out = DenseK(1, name="min_head", dtype="float32")(shared)
    model = Model(inp, {"hour_head": hour_out, "min_head": min_out}, name="multi_regular")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={"hour_head": "sparse_categorical_crossentropy", "min_head": "mse"},
        loss_weights={"hour_head": 1.0, "min_head": minute_loss_weight},
        metrics={"hour_head": ["accuracy"], "min_head": ["mae"]}
    )
    return model

def build_head_circle(backbone, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    x = DenseK(256, activation="relu")(z)
    x = L.Dropout(DROPOUT)(x)
    out = DenseK(2, activation="linear", name="time_angle_vec", dtype="float32")(x)  # fp32 head
    model = Model(inp, out, name="circle_time")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_head_multi_circle(backbone, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    shared = DenseK(256, activation="relu", name="mh_shared_dense")(z)
    shared = L.Dropout(DROPOUT, name="mh_shared_do")(shared)
    hour_logits = DenseK(12, name="hour_logits")(shared)
    hour_out = L.Activation("softmax", name="hour_head")(hour_logits)
    min_out = DenseK(2, activation="linear", name="min_head", dtype="float32")(shared)  # fp32 head
    model = Model(inp, {"hour_head": hour_out, "min_head": min_out}, name="multi_circle")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={"hour_head": "sparse_categorical_crossentropy", "min_head": "mse"},
        metrics={"hour_head": "accuracy", "min_head": "mae"}
    )
    return model

def get_custom_backbone(name):
    if name == "light":
        return get_backbone_light(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)
    if name == "heavy":
        return get_backbone_heavy(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)
    return get_backbone_medium(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)

def angle_targets_from_labels(y_hm):
    mins_total = minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32) % 720.0
    theta = mins_total / 720.0 * (2.0 * np.pi)
    sin_theta = np.sin(theta).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)
    return np.stack([sin_theta, cos_theta], axis=1)

def split_80_20():
    N = len(X)
    idx = np.arange(N)
    np.random.seed(1)
    np.random.shuffle(idx)
    a = int(0.8 * N)
    return idx[:a], idx[a:]

def binning(n_bins):
    bin_size = 720.0 / n_bins
    bins = np.floor(minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32) / bin_size).astype(np.int32) % n_bins
    bin_centers = (np.arange(n_bins) * bin_size + bin_size / 2.0) % 720.0
    return bins, bin_centers

# -------------------- CSV logging callback --------------------
def extract_lr(opt):
    try:
        return float(tf.keras.backend.get_value(opt.learning_rate))
    except Exception:
        return None

def acc_within_threshold(cse_vec, thr_min):
    cse = np.asarray(cse_vec, dtype=np.float32)
    return float(np.mean(cse <= float(thr_min)))

class EpochCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, key, train_xy, val_xy, cse_fn,
                 csv_path=EPOCH_CSV_PATH, thresholds=TOL_THRESHOLDS):
        super().__init__()
        self.key = key
        self.train_xy = train_xy
        self.val_xy = val_xy
        self.cse_fn = cse_fn
        self.thresholds = list(thresholds)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = None

        if self.key not in epoch_stats:
            epoch_stats[self.key] = {"rows": [], "time": [], "cse": [], "val_loss": []}

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def _summarize_cse(self, cse):
        cse = np.asarray(cse, dtype=np.float32)
        out = {
            "cse_mean": float(np.mean(cse)),
            "cse_median": float(np.median(cse)),
            "cse_p90": float(np.percentile(cse, 90)),
        }
        for t in self.thresholds:
            out[f"acc_le_{int(t)}m"] = acc_within_threshold(cse, t)
        return out
    
    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self.t0 if self.t0 else 0.0

        (Xtr, ytr) = self.train_xy
        (Xva, yva) = self.val_xy

        cse_tr = self.cse_fn(self.model, Xtr, ytr)
        cse_va = self.cse_fn(self.model, Xva, yva)

        row = {c: None for c in ALL_COLS}
        row.update({
            "model": self.key,
            "epoch": int(epoch + 1),
            "time_s": float(dt),
            "lr": extract_lr(self.model.optimizer),
        })

        for k, v in (logs or {}).items():
            if k in row:
                row[k] = float(v)

        def summarize_cse(cse):
            cse = np.asarray(cse, dtype=np.float32)
            d = {
                "cse_mean": float(np.mean(cse)),
                "cse_median": float(np.median(cse)),
                "cse_p90": float(np.percentile(cse, 90)),
                "acc_le_1m": float(np.mean(cse <= 1.0)),
                "acc_le_5m": float(np.mean(cse <= 5.0)),
                "acc_le_10m": float(np.mean(cse <= 10.0)),
            }
            return d

        tr = summarize_cse(cse_tr)
        va = summarize_cse(cse_va)

        row["train_cse_mean"]   = tr["cse_mean"]
        row["train_cse_median"] = tr["cse_median"]
        row["train_cse_p90"]    = tr["cse_p90"]
        row["train_acc_le_1m"]  = tr["acc_le_1m"]
        row["train_acc_le_5m"]  = tr["acc_le_5m"]
        row["train_acc_le_10m"] = tr["acc_le_10m"]

        row["val_cse_mean"]     = va["cse_mean"]
        row["val_cse_median"]   = va["cse_median"]
        row["val_cse_p90"]      = va["cse_p90"]
        row["val_acc_le_1m"]    = va["acc_le_1m"]
        row["val_acc_le_5m"]    = va["acc_le_5m"]
        row["val_acc_le_10m"]   = va["acc_le_10m"]

        epoch_stats[self.key]["rows"].append(row)
        epoch_stats[self.key]["time"].append(float(dt))
        epoch_stats[self.key]["cse"].append(row["val_cse_mean"])
        epoch_stats[self.key]["val_loss"].append(row.get("val_loss", None))

        df_row = pd.DataFrame([row], columns=ALL_COLS)
        header = not self.csv_path.exists()
        df_row.to_csv(self.csv_path, mode="a", header=header, index=False)

# -------------------- Helpers for summaries --------------------
def describe_errors(name, errs_min):
    print(80 * "==")
    print(name.upper())
    print({
        "mean_cse_min": float(np.mean(errs_min)),
        "median_cse_min": float(np.median(errs_min)),
        "p90_cse_min": float(np.percentile(errs_min, 90))
    })
    print(80 * "==")

def log_result(name, cse_arr):
    stats = {
        "model": name,
        "mean_cse": float(np.mean(cse_arr)),
        "median_cse": float(np.median(cse_arr)),
        "p90_cse": float(np.percentile(cse_arr, 90))
    }
    results_summary.append(stats)
    print(stats)

def epoch_stats_to_df(epoch_stats):
    rows = []
    for _, d in epoch_stats.items():
        rows.extend(d.get("rows", []))
    return pd.DataFrame(rows)

def save_tables_for_report(epoch_stats, results_summary, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    df_epoch = epoch_stats_to_df(epoch_stats)
    if not df_epoch.empty:
        df_epoch.to_csv(out / "epoch_stats_long.csv", index=False)
    if len(results_summary):
        pd.DataFrame(results_summary).to_csv(out / "results_summary.csv", index=False)

def add_eval(cbs, cb):
    return list(cbs) + [cb]

# -------------------- Training orchestration --------------------
def get_custom_cbs(key):
    return [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, monitor="val_loss")
    ]

def run_all(backbones, label_representations, bins):
    tr_all, te_idx = split_80_20()

    rng = np.random.default_rng(SEED)
    tr_shuf = tr_all.copy()
    rng.shuffle(tr_shuf)
    val_size = max(1, int(round(0.125 * len(tr_shuf))))
    va_idx = tr_shuf[:val_size]
    tr_idx = tr_shuf[val_size:]

    for bname in backbones:

        if "clf" in label_representations:
            for nb in bins:
                try:
                    bb = get_custom_backbone(bname)
                    y_bins, centers = binning(nb)

                    ds_tr = make_ds_classification(X[tr_idx], y_bins[tr_idx], shuffle=True)
                    ds_va = make_ds_classification(X[va_idx], y_bins[va_idx])

                    clf = build_head_classification(bb, num_classes=nb, lr=1e-3)
                    key = f"{bname}__classifier_{nb}bins"

                    cb_csv = EpochCSVLogger(
                        key=key,
                        train_xy=(X[tr_idx], y_hm[tr_idx]),
                        val_xy=(X[va_idx], y_hm[va_idx]),
                        cse_fn=lambda m, Xt, Yt: _cse_clf_bins(m, Xt, Yt, centers)
                    )

                    cbs = add_eval(get_custom_cbs(key), cb_csv)

                    clf.fit(
                        ds_tr,
                        validation_data=ds_va,
                        epochs=MAX_EPOCHS,
                        callbacks=cbs,
                        verbose=2,
                    )

                    cse = _cse_clf_bins(clf, X[te_idx], y_hm[te_idx], centers)
                    describe_errors(key, cse)
                    log_result(key, cse)

                finally:
                    for obj in ("ds_tr", "ds_va", "clf", "bb"):
                        try:
                            del locals()[obj]
                        except Exception:
                            pass
                    clear_tf_session()

        if "reg" in label_representations:
            try:
                bb = get_custom_backbone(bname)
                y_reg = (y_hm[:, 0].astype(np.float32) % 12.0) + (y_hm[:, 1].astype(np.float32) / 60.0)

                ds_tr = make_ds_regression(X[tr_idx], y_reg[tr_idx], shuffle=True)
                ds_va = make_ds_regression(X[va_idx], y_reg[va_idx])

                reg = build_head_regression(bb, lr=1e-3)
                key = f"{bname}__regression"

                cb_csv = EpochCSVLogger(
                    key=key,
                    train_xy=(X[tr_idx], y_hm[tr_idx]),
                    val_xy=(X[va_idx], y_hm[va_idx]),
                    cse_fn=_cse_reg
                )

                cbs = add_eval(get_custom_cbs(key), cb_csv)

                reg.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=cbs,
                    verbose=2,
                )

                cse = _cse_reg(reg, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)

            finally:
                for obj in ("ds_tr", "ds_va", "reg", "bb"):
                    try:
                        del locals()[obj]
                    except Exception:
                        pass
                clear_tf_session()

        if "circle" in label_representations:
            try:
                bb = get_custom_backbone(bname)

                ds_tr = make_ds_circle(X[tr_idx], y_hm[tr_idx], shuffle=True)
                ds_va = make_ds_circle(X[va_idx], y_hm[va_idx])

                circ = build_head_circle(bb, lr=1e-3)
                key = f"{bname}__circle"

                cb_csv = EpochCSVLogger(
                    key=key,
                    train_xy=(X[tr_idx], y_hm[tr_idx]),
                    val_xy=(X[va_idx], y_hm[va_idx]),
                    cse_fn=_cse_circle
                )

                cbs = add_eval(get_custom_cbs(key), cb_csv)

                circ.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=cbs,
                    verbose=2,
                )

                cse = _cse_circle(circ, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)

            finally:
                for obj in ("ds_tr", "ds_va", "circ", "bb"):
                    try:
                        del locals()[obj]
                    except Exception:
                        pass
                clear_tf_session()

        if "multi_regular" in label_representations:
            try:
                bb = get_custom_backbone(bname)

                ds_tr = make_ds_multi(X[tr_idx], y_hm[tr_idx], shuffle=True)
                ds_va = make_ds_multi(X[va_idx], y_hm[va_idx])

                mh_reg = build_head_multi(bb, lr=1e-3)
                key = f"{bname}__multi_regular"

                cb_csv = EpochCSVLogger(
                    key=key,
                    train_xy=(X[tr_idx], y_hm[tr_idx]),
                    val_xy=(X[va_idx], y_hm[va_idx]),
                    cse_fn=lambda m, Xt, Yt: eval_common_sense_error_multi_regular(m, Xt, Yt)
                )

                cbs = add_eval(get_custom_cbs(key), cb_csv)

                mh_reg.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=cbs,
                    verbose=2,
                )

                cse = eval_common_sense_error_multi_regular(mh_reg, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)

            finally:
                for obj in ("ds_tr", "ds_va", "mh_reg", "bb"):
                    try:
                        del locals()[obj]
                    except Exception:
                        pass
                clear_tf_session()

        if "multi_circle" in label_representations:
            try:
                bb = get_custom_backbone(bname)

                ds_tr = make_ds_multi_circle(X[tr_idx], y_hm[tr_idx], shuffle=True)
                ds_va = make_ds_multi_circle(X[va_idx], y_hm[va_idx])

                mh_circ = build_head_multi_circle(bb, lr=1e-3)
                key = f"{bname}__multi_circle"

                cb_csv = EpochCSVLogger(
                    key=key,
                    train_xy=(X[tr_idx], y_hm[tr_idx]),
                    val_xy=(X[va_idx], y_hm[va_idx]),
                    cse_fn=lambda m, Xt, Yt: eval_common_sense_error_multi_circle(m, Xt, Yt)
                )

                cbs = add_eval(get_custom_cbs(key), cb_csv)

                mh_circ.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=cbs,
                    verbose=2,
                )

                cse = eval_common_sense_error_multi_circle(mh_circ, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)

            finally:
                for obj in ("ds_tr", "ds_va", "mh_circ", "bb"):
                    try:
                        del locals()[obj]
                    except Exception:
                        pass
                clear_tf_session()

    save_tables_for_report(epoch_stats, results_summary)

# -------------------- Load data & run --------------------
X, y_hm = load_clock_data(DATA_DIR)
train_idx, val_idx, test_idx = split_indices(len(X), train=0.8, val=0.1, seed=SEED)
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y_hm[train_idx], y_hm[val_idx], y_hm[test_idx]

def main():
    backbones = ["light", "medium", "heavy"]
    # Five label representations / heads:
    # - classifier (bins = [24, 144, 720])
    # - regression
    # - multi-head-regular
    # - circle
    # - multi-head-circle
    label_representations = ["clf", "reg", "multi_regular", "circle", "multi_circle"]
    bins = [24, 144, 720]
    run_all(backbones, label_representations, bins)

if __name__ == "__main__":
    main()
