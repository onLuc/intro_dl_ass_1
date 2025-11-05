import time
import gc
import numpy as np

import seaborn as sns
import pandas as pd

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.pop("TF_GPU_ALLOCATOR", None)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras import mixed_precision as mp

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

mp.set_global_policy("mixed_float16")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("../IDL_A1_task_2_runs/data")
ARTIFACTS_DIR = Path("./artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
MAX_EPOCHS = 50
CHECKPOINT_INTERVAL = 15

INITIALIZER = "he_uniform"
DROPOUT = 0.3

epoch_stats = {}


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


class IntervalCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, interval=15, out_dir=ARTIFACTS_DIR, prefix="clf24"):
        super().__init__()
        self.interval = interval
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            fname = f"{self.prefix}_epoch{epoch + 1:03d}.weights.h5"
            path = str(self.out_dir / fname)
            self.model.save_weights(path)
            print(f"[Checkpoint] Saved weights at epoch {epoch + 1} -> {path}")


class EvalTracker(tf.keras.callbacks.Callback):
    def __init__(self, key, X, y, cse_fn):
        super().__init__()
        self.key = key
        self.X = X
        self.y = y
        self.cse_fn = cse_fn
        if self.key not in epoch_stats:
            epoch_stats[self.key] = {"time": [], "cse": [], "val_loss": []}
        self.t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self.t0 if self.t0 else 0.0
        cse = self.cse_fn(self.model, self.X, self.y)

        d = epoch_stats[self.key]
        d["time"].append(float(dt))
        d["cse"].append(float(np.mean(cse)))
        # capture val_loss if provided by Keras (when validation_data is used)
        if logs is not None and "val_loss" in logs:
            d["val_loss"].append(float(logs["val_loss"]))
        else:
            d["val_loss"].append(None)

        print(f"epoch {epoch + 1} mean CSE {np.mean(cse):.2f} min")


X, y_hm = load_clock_data(DATA_DIR)
train_idx, val_idx, test_idx = split_indices(len(X), train=0.8, val=0.1, seed=SEED)
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y_hm[train_idx], y_hm[val_idx], y_hm[test_idx]

results_summary = []


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
    vecs = model.predict(X_sub[..., None], verbose=0)
    pred_total_min = []
    for (s, c) in vecs:
        pred_total_min.append(decode_circle(s, c))
    pred_total_min = np.array(pred_total_min, dtype=np.float32)
    true_total_min = (minutes_after_midnight(y_hm_sub[:, 0], y_hm_sub[:, 1]) % 720).astype(np.float32)
    cse = np.array([circular_minute_distance(t, p) for t, p in zip(true_total_min, pred_total_min)])
    return cse, pred_total_min


def _cse_circle(model, X, y):
    cse, _ = eval_common_sense_error_circle(model, X, y)
    return cse


def _cse_reg(model, X, y):
    cse, _ = eval_common_sense_error_regressor(model, X, y)
    return cse


def _cse_clf_bins(model, X, y, bin_centers):
    y_pred = model.predict(X[..., None], verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)
    true_min = minutes_after_midnight(y[:, 0], y[:, 1]).astype(np.float32)
    pred_min = bin_centers[y_pred_cls].astype(np.float32)
    diff = np.abs(pred_min - true_min)
    return np.minimum(diff, 720.0 - diff)


def add_eval(cbs, cb):
    return list(cbs) + [cb]


def get_backbone_light(input_shape=(75, 75, 1), dropout=DROPOUT
                       ):
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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_head_regression(backbone, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    x = DenseK(256, activation="relu")(z)
    x = L.Dropout(DROPOUT)(x)
    out = DenseK(1, name="time_float")(x)
    model = Model(inp, out, name="regressor_time")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def minutes_to_unit_vec_60(mins):
    theta = (mins % 60) / 60.0 * (2.0 * np.pi)
    return np.stack([np.sin(theta), np.cos(theta)], axis=1).astype(np.float32)


def make_ds_classification(X, y_bins, shuffle=False):
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, y_bins.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(1)


def make_regression_targets(y_hm):
    y_reg = (y_hm[:, 0] % 12).astype(np.float32) + (y_hm[:, 1].astype(np.float32) / 60.0)
    return y_reg


def make_ds_regression(X, y_reg, shuffle=False):
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, y_reg.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(1)


def make_ds_multi(X, y_hm, shuffle=False):
    y_hours = (y_hm[:, 0] % 12).astype(np.int32)
    y_minutes = y_hm[:, 1].astype(np.float32)
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, {"hour_head": y_hours, "min_head": y_minutes}))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(1)


def make_ds_circle(X, y_hm, shuffle=False):
    y_circle = angle_targets_from_labels(y_hm)
    Xc = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((Xc, y_circle.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(10_000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(1)


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


def build_head_multi_circle(backbone, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    shared = DenseK(256, activation="relu", name="mh_shared_dense")(z)
    shared = L.Dropout(DROPOUT, name="mh_shared_do")(shared)
    hour_logits = DenseK(12, name="hour_logits")(shared)
    hour_out = L.Activation("softmax", name="hour_head")(hour_logits)
    min_out = DenseK(2, activation="linear", name="min_head")(shared)
    model = Model(inp, {"hour_head": hour_out, "min_head": min_out}, name="multi_circle")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={"hour_head": "sparse_categorical_crossentropy", "min_head": "mse"},
                  metrics={"hour_head": "accuracy", "min_head": "mae"})
    return model


def eval_common_sense_error_multi_circle(model, X, y_hm):
    preds = model.predict(X[..., None], verbose=0)
    y_hour = np.argmax(preds["hour_head"], axis=1)
    v = preds["min_head"].astype(np.float32)
    ang = np.arctan2(v[:, 0], v[:, 1])
    ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)
    min_pred = (ang / (2.0 * np.pi) * 60.0)
    total_pred = (y_hour.astype(np.float32) * 60.0 + min_pred) % 720.0
    true_min = minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32)
    diff = np.abs(total_pred - true_min)
    return np.minimum(diff, 720.0 - diff)


def build_head_multihead(backbone, lr=1e-3, minute_loss_weight=1.0):
    inp = backbone.input
    z = backbone.output
    shared = DenseK(256, activation="relu", name="mh_shared_dense")(z)
    shared = L.Dropout(DROPOUT, name="mh_shared_do")(shared)
    hour_logits = DenseK(12, name="hour_logits")(shared)
    hour_out = L.Activation("softmax", name="hour_head")(hour_logits)
    min_out = DenseK(1, name="min_head")(shared)
    model = Model(inp, {"hour_head": hour_out, "min_head": min_out}, name="multihead_time")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={"hour_head": "sparse_categorical_crossentropy", "min_head": "mse"},
                  loss_weights={"hour_head": 1.0, "min_head": minute_loss_weight},
                  metrics={"hour_head": ["accuracy"], "min_head": ["mae"]})
    return model


def angle_targets_from_labels(y_hm):
    mins_total = minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32) % 720.0
    theta = mins_total / 720.0 * (2.0 * np.pi)
    sin_theta = np.sin(theta).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)
    return np.stack([sin_theta, cos_theta], axis=1)


def split_80_10_10():
    N = len(X)
    idx = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(idx)
    a = int(0.8 * N)
    b = int(0.9 * N)
    return idx[:a], idx[a:b], idx[b:]


def split_80_20():
    N = len(X)
    idx = np.arange(N)
    np.random.seed(1)
    np.random.shuffle(idx)
    a = int(0.8 * N)
    return idx[:a], idx[a:]


def build_head_circle(backbone, lr=1e-3):
    inp = backbone.input
    z = backbone.output
    x = DenseK(256, activation="relu")(z)
    x = L.Dropout(DROPOUT)(x)
    out = DenseK(2, activation="linear", name="time_angle_vec")(x)
    model = Model(inp, out, name="circle_time")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def get_custom_backbone(name):
    if name == "light":
        return get_backbone_light(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)
    if name == "heavy":
        return get_backbone_heavy(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)
    return get_backbone_medium(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)


def binning(n_bins):
    bin_size = 720.0 / n_bins
    bins = np.floor(minutes_after_midnight(y_hm[:, 0], y_hm[:, 1]).astype(np.float32) / bin_size).astype(
        np.int32) % n_bins
    bin_centers = (np.arange(n_bins) * bin_size + bin_size / 2.0) % 720.0
    return bins, bin_centers


def describe_errors(name, errs_min):
    print(80 * "==")
    print(name.upper())
    print({
        "mean_cse_min": float(np.mean(errs_min)),
        "median_cse_min": float(np.median(errs_min)),
        "p90_cse_min": float(np.percentile(errs_min, 90))
    })
    print(80 * "==")


def save_tables_for_report(epoch_stats, results_summary, out_dir="plots"):
    out = Path(out_dir);
    out.mkdir(exist_ok=True)
    # per-epoch long table
    df_epoch = summarize_results(epoch_stats)
    df_epoch.to_csv(out / "epoch_stats_long.csv", index=False)

    # per-model final summary of CSE
    if len(results_summary):
        df_sum = pd.DataFrame(results_summary)
        df_sum.to_csv(out / "results_summary.csv", index=False)


def log_result(name, cse_arr):
    stats = {"model": name, "mean_cse": float(np.mean(cse_arr)), "median_cse": float(np.median(cse_arr)),
             "p90_cse": float(np.percentile(cse_arr, 90))}
    results_summary.append(stats)
    print(stats)


def summarize_results(epoch_stats):
    """Collect flattened stats for plotting"""
    rows = []
    for model_name, d in epoch_stats.items():
        for e, (t, c) in enumerate(zip(d.get("time", []), d.get("cse", []))):
            rows.append({
                "model": model_name,
                "epoch": e + 1,
                "time": t,
                "cse": c,
                "type": (
                    "classification" if "clf" in model_name else
                    "regression" if "reg" in model_name else
                    "circle" if "circle" in model_name and "multi" not in model_name else
                    "multi-circle"
                ),
                "backbone": (
                    "light" if "light" in model_name else
                    "heavy" if "heavy" in model_name else
                    "medium"
                )
            })
    return pd.DataFrame(rows)


def plot_run_all_summaries(epoch_stats, results_summary):
    df = summarize_results(epoch_stats)
    Path("plots").mkdir(exist_ok=True)

    # 1) Training time vs backbone (avg per representation)
    plt.figure(figsize=(8, 5))
    g = df.groupby(["backbone", "type"])["time"].mean().reset_index()
    sns.barplot(data=g, x="type", y="time", hue="backbone")  # no palette -> no future warning
    plt.title("Average Training Time per Model and Backbone")
    plt.ylabel("Time per Epoch (s)")
    plt.tight_layout()
    plt.savefig("plots/training_time_vs_backbone.png", dpi=150)

    # 2) CSE per epoch (all models)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="cse", hue="model", linewidth=1.0, legend=True)
    plt.title("Common Sense Error per Epoch")
    plt.ylabel("Mean CSE (minutes)")
    plt.tight_layout()
    plt.savefig("plots/cse_per_epoch_all_models.png", dpi=150)

    # 3) Validation loss per model (build a tidy DF from epoch_stats)
    rows_vl = []
    for m, s in epoch_stats.items():
        vls = s.get("val_loss", [])
        for i, v in enumerate(vls, start=1):
            if v is not None:
                rows_vl.append({"model": m, "epoch": i, "val_loss": v})
    plt.figure(figsize=(10, 6))
    if rows_vl:
        df_vl = pd.DataFrame(rows_vl)
        sns.lineplot(data=df_vl, x="epoch", y="val_loss", hue="model", linewidth=0.9)
    else:
        plt.text(0.5, 0.5, "No val_loss logged", ha="center", va="center")
        plt.axis("off")
    plt.title("Validation Loss over Epochs")
    plt.tight_layout()
    plt.savefig("plots/val_loss_per_model.png", dpi=150)

    # 4) Mean CSE by representation (use hue to silence warning)
    plt.figure(figsize=(8, 5))
    mean_cse = df.groupby("type")["cse"].mean().reset_index()
    sns.barplot(data=mean_cse, x="type", y="cse", hue="type", dodge=False, legend=False)
    plt.title("Mean CSE across Model Representations")
    plt.ylabel("Mean CSE (minutes)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("plots/mean_cse_by_representation.png", dpi=150)

    # 5) Optional: mean CSE vs backbone (accuracy vs capacity)
    plt.figure(figsize=(8, 5))
    m2 = df.groupby(["backbone"])["cse"].mean().reset_index()
    sns.barplot(data=m2, x="backbone", y="cse", hue="backbone", dodge=False, legend=False)
    plt.title("Mean CSE by Backbone")
    plt.ylabel("Mean CSE (minutes)")
    plt.tight_layout()
    plt.savefig("plots/mean_cse_by_backbone.png", dpi=150)

    # Save tables for report
    save_tables_for_report(epoch_stats, results_summary)

    print("✅ Saved plots + tables in ./plots/")


def plot_final_80_20_curves():
    # draw CSE vs epoch for models with 'final' in name
    final_keys = [k for k in epoch_stats.keys() if "final" in k]
    plt.figure(figsize=(9, 5))
    any_line = False
    for k in sorted(final_keys):
        d = epoch_stats[k].get("cse", [])
        if d:
            xs = list(range(1, len(d) + 1))
            plt.plot(xs, d, label=k, linewidth=1.2)
            any_line = True
    if any_line:
        plt.legend(fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Mean CSE (minutes)")
        plt.title("Final 80/20: CSE over Epochs")
    else:
        plt.axis("off")
        plt.text(0.5, 0.5, "No final 80/20 traces found", ha="center", va="center")
    plt.tight_layout()
    plt.savefig("plots/final_80_20.png", dpi=150)


def plot_final_histograms(cse_circle, cse_multi):
    Path("plots").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    if cse_circle is not None:
        plt.hist(cse_circle, bins=40, alpha=0.6, label="final_circle")
    if cse_multi is not None:
        plt.hist(cse_multi, bins=40, alpha=0.6, label="final_multi_circle")
    plt.title("Final 80/20: CSE Distribution (Test Split)")
    plt.xlabel("CSE (minutes)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/final_80_20_cse_distribution.png", dpi=150)


def run_final_compare():
    tr, te = split_80_20()

    # --- final circle ---
    bb1 = get_backbone_medium(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)
    ds_tr = make_ds_circle(X[tr], y_hm[tr], shuffle=True)
    ds_te = make_ds_circle(X[te], y_hm[te])
    c1 = build_head_circle(bb1, lr=1e-3)
    key1 = "medium_final_circle"
    cbs1 = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
            IntervalCheckpoint(interval=CHECKPOINT_INTERVAL, out_dir=ARTIFACTS_DIR, prefix=key1)]
    c1.fit(ds_tr, validation_data=ds_te, epochs=MAX_EPOCHS,
           callbacks=add_eval(cbs1, EvalTracker(key1, X[te], y_hm[te], _cse_circle)),
           verbose=2)
    cse1, _ = eval_common_sense_error_circle(c1, X[te], y_hm[te])

    # --- final multi-circle ---
    bb2 = get_backbone_medium(input_shape=IMAGE_SIZE + (1,), dropout=DROPOUT)
    ds_tr2 = make_ds_multi_circle(X[tr], y_hm[tr], shuffle=True)
    ds_te2 = make_ds_multi_circle(X[te], y_hm[te])
    m2 = build_head_multi_circle(bb2, lr=1e-3)
    key2 = "medium_final_multi_circle"
    cbs2 = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
            IntervalCheckpoint(interval=CHECKPOINT_INTERVAL, out_dir=ARTIFACTS_DIR, prefix=key2)]
    m2.fit(ds_tr2, validation_data=ds_te2, epochs=MAX_EPOCHS,
           callbacks=add_eval(cbs2, EvalTracker(key2, X[te], y_hm[te],
                                                lambda m, Xt, Yt: eval_common_sense_error_multi_circle(m, Xt, Yt))),
           verbose=2)
    cse2 = eval_common_sense_error_multi_circle(m2, X[te], y_hm[te])

    # Plots for the final section
    plot_final_80_20_curves()
    plot_final_histograms(cse1, cse2)


def run_all(backbones, label_representations, bins):
    tr_idx, va_idx, te_idx = split_80_10_10()

    for bname in backbones:

        if "clf" in label_representations:
            for nb in bins:
                try:
                    bb = get_custom_backbone(bname)
                    y_bins, centers = binning(nb)

                    ds_tr = make_ds_classification(X[tr_idx], y_bins[tr_idx], shuffle=True)
                    ds_va = make_ds_classification(X[va_idx], y_bins[va_idx])

                    clf = build_head_classification(bb, num_classes=nb, lr=1e-3)
                    key = f"{bname}_clf{nb}"
                    cbs = [
                        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, monitor="val_loss"),
                        IntervalCheckpoint(interval=CHECKPOINT_INTERVAL, out_dir=ARTIFACTS_DIR, prefix=key),
                    ]

                    clf.fit(
                        ds_tr,
                        validation_data=ds_va,
                        epochs=MAX_EPOCHS,
                        callbacks=add_eval(
                            cbs,
                            EvalTracker(
                                key, X[te_idx], y_hm[te_idx],
                                lambda m, Xt, Yt: _cse_clf_bins(m, Xt, Yt, centers)
                            )
                        ),
                        verbose=2,
                    )

                    cse = _cse_clf_bins(clf, X[te_idx], y_hm[te_idx], centers)
                    describe_errors(key, cse)
                    log_result(key, cse)


                finally:
                    # release references before clearing TF context
                    try:
                        del ds_tr
                    except:
                        pass
                    try:
                        del ds_va
                    except:
                        pass
                    try:
                        del clf
                    except:
                        pass
                    try:
                        del bb
                    except:
                        pass
                    clear_tf_session()

        if "reg" in label_representations:
            try:
                bb = get_custom_backbone(bname)
                y_reg = (y_hm[:, 0].astype(np.float32) % 12.0) + (y_hm[:, 1].astype(np.float32) / 60.0)

                ds_tr = make_ds_regression(X[tr_idx], y_reg[tr_idx], shuffle=True)
                ds_va = make_ds_regression(X[va_idx], y_reg[va_idx])

                reg = build_head_regression(bb, lr=1e-3)
                key = f"{bname}_reg"
                cbs = [
                    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, monitor="val_loss"),
                    IntervalCheckpoint(interval=CHECKPOINT_INTERVAL, out_dir=ARTIFACTS_DIR, prefix=key),
                ]

                reg.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=add_eval(cbs, EvalTracker(key, X[te_idx], y_hm[te_idx], _cse_reg)),
                    verbose=2,
                )

                cse = _cse_reg(reg, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)


            finally:
                try:
                    del ds_tr
                except:
                    pass
                try:
                    del ds_va
                except:
                    pass
                try:
                    del reg
                except:
                    pass
                try:
                    del bb
                except:
                    pass
                clear_tf_session()

        if "circle" in label_representations:
            try:
                bb = get_custom_backbone(bname)

                ds_tr = make_ds_circle(X[tr_idx], y_hm[tr_idx], shuffle=True)
                ds_va = make_ds_circle(X[va_idx], y_hm[va_idx])

                circ = build_head_circle(bb, lr=1e-3)
                key = f"{bname}_circle"
                cbs = [
                    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, monitor="val_loss"),
                    IntervalCheckpoint(interval=CHECKPOINT_INTERVAL, out_dir=ARTIFACTS_DIR, prefix=key),
                ]

                circ.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=add_eval(cbs, EvalTracker(key, X[te_idx], y_hm[te_idx], _cse_circle)),
                    verbose=2,
                )

                cse = _cse_circle(circ, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)


            finally:
                try:
                    del ds_tr
                except:
                    pass
                try:
                    del ds_va
                except:
                    pass
                try:
                    del circ
                except:
                    pass
                try:
                    del bb
                except:
                    pass
                clear_tf_session()

        if "multi" in label_representations:
            try:
                bb = get_custom_backbone(bname)

                ds_tr = make_ds_multi_circle(X[tr_idx], y_hm[tr_idx], shuffle=True)
                ds_va = make_ds_multi_circle(X[va_idx], y_hm[va_idx])

                mh = build_head_multi_circle(bb, lr=1e-3)
                key = f"{bname}_multi_circle"
                cbs = [
                    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6, monitor="val_loss"),
                    IntervalCheckpoint(interval=CHECKPOINT_INTERVAL, out_dir=ARTIFACTS_DIR, prefix=key),
                ]

                mh.fit(
                    ds_tr,
                    validation_data=ds_va,
                    epochs=MAX_EPOCHS,
                    callbacks=add_eval(
                        cbs,
                        EvalTracker(
                            key, X[te_idx], y_hm[te_idx],
                            lambda m, Xt, Yt: eval_common_sense_error_multi_circle(m, Xt, Yt)
                        )
                    ),
                    verbose=2,
                )

                cse = eval_common_sense_error_multi_circle(mh, X[te_idx], y_hm[te_idx])
                describe_errors(key, cse)
                log_result(key, cse)


            finally:
                try:
                    del ds_tr
                except:
                    pass
                try:
                    del ds_va
                except:
                    pass
                try:
                    del mh
                except:
                    pass
                try:
                    del bb
                except:
                    pass
                clear_tf_session()

    plot_run_all_summaries(epoch_stats, results_summary)


def main():
    backbones = ["light", "medium", "heavy"]
    label_representations = ["clf", "reg", "multi", "circle"]
    bins = [24, 144, 720]

    # backbones = ["light", "medium"]
    # label_representations = ["clf", "circle"]
    # bins = [24]

    run_all(backbones, label_representations, bins)


if __name__ == "__main__":
    main()
