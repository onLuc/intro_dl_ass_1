# task1b_cifar10_transfer.py
"""
Task 1(b): Transfer best-performing Fashion-MNIST configurations to CIFAR-10
This script:
  - Loads top 3 configs from mlp_results_summary.csv and cnn_results_summary.csv
  - Rebuilds those models with the same hyperparameters
  - Trains and evaluates them on the CIFAR-10 dataset
  - Saves results and training plots
"""

import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers # type: ignore
import matplotlib.pyplot as plt

# ==========================================================
# Utilities
# ==========================================================

def build_mlp_model(input_shape, activation, initializer, regularizer, dropout_rate):
    reg = None
    if regularizer == "l1":
        reg = regularizers.l1(0.001)
    elif regularizer == "l2":
        reg = regularizers.l2(0.001)

    if dropout_rate != dropout_rate:  # NaN check
        dropout_rate = None

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(300, activation=activation,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=reg))
    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(100, activation=activation,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=reg))
    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


def build_cnn_model(input_shape, activation, initializer, regularizer, dropout_rate):
    reg = None
    if regularizer == "l1":
        reg = regularizers.l1(0.001)
    elif regularizer == "l2":
        reg = regularizers.l2(0.001)

    if dropout_rate != dropout_rate:  # NaN check
        dropout_rate = None

    model = keras.Sequential([
        keras.layers.Conv2D(64, 7, activation=activation, padding="same",
                            kernel_initializer=initializer, kernel_regularizer=reg,
                            input_shape=input_shape),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation=activation, padding="same",
                            kernel_initializer=initializer, kernel_regularizer=reg),
        keras.layers.Conv2D(128, 3, activation=activation, padding="same",
                            kernel_initializer=initializer, kernel_regularizer=reg),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation=activation, padding="same",
                            kernel_initializer=initializer, kernel_regularizer=reg),
        keras.layers.Conv2D(256, 3, activation=activation, padding="same",
                            kernel_initializer=initializer, kernel_regularizer=reg),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=activation,
                           kernel_initializer=initializer, kernel_regularizer=reg)
    ])
    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(64, activation=activation,
                                 kernel_initializer=initializer, kernel_regularizer=reg))
    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


def get_optimizer(name):
    mapping = {
        "adam_good": keras.optimizers.Adam(learning_rate=0.001),
        "adam_bad": keras.optimizers.Adam(learning_rate=0.05),
        "sgd_good": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "sgd_bad": keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0)
    }
    return mapping.get(name, keras.optimizers.Adam(0.001))


def train_and_evaluate(model, optimizer, x_train, y_train, x_test, y_test,
                       tag, out_dir="src/task_1/plots/cifar10"):
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    os.makedirs(out_dir, exist_ok=True)

    # Save training plot
    plt.figure()
    plt.plot(hist.history["accuracy"], label="Train acc")
    plt.plot(hist.history["val_accuracy"], label="Val acc")
    plt.title(f"{tag} (CIFAR-10)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, f"{tag}.png")
    plt.savefig(path)
    plt.close()

    return test_acc, test_loss, path


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.squeeze(), y_test.squeeze()

    # Load previous experiment summaries
    mlp_df = pd.read_csv("src/task_1/mlp_results_summary.csv").sort_values(by="test_acc", ascending=False).head(3)
    cnn_df = pd.read_csv("src/task_1/cnn_results_summary.csv").sort_values(by="test_acc", ascending=False).head(3)

    results = []

    for model_type, df in [("mlp", mlp_df), ("cnn", cnn_df)]:
        for i, row in df.iterrows():
            act, init, opt_name, reg, drop = row["activation"], row["initializer"], row["optimizer"], row["regularizer"], row["dropout"]
            optimizer = get_optimizer(opt_name)
            tag = f"{model_type}_top{i}_{act}_{init}_{opt_name}_{reg}_drop{drop}"

            if model_type == "mlp":
                model = build_mlp_model((32, 32, 3), act, init, reg, drop)
                xtr, xte = x_train, x_test
            else:
                model = build_cnn_model((32, 32, 3), act, init, reg, drop)
                xtr, xte = x_train, x_test

            acc, loss, plot_path = train_and_evaluate(model, optimizer, xtr, y_train, xte, y_test, tag)
            print(f"✅ {model_type.upper()} ({tag}) — Test acc: {acc:.4f}, loss: {loss:.4f}")
            results.append({
                "model": model_type,
                "activation": act,
                "initializer": init,
                "optimizer": opt_name,
                "regularizer": reg,
                "dropout": drop,
                "test_acc": acc,
                "test_loss": loss,
                "plot": plot_path
            })

    # Save summary
    out_df = pd.DataFrame(results)
    out_df.to_csv("cifar10_transfer_results.csv", index=False)
    print("\nAll CIFAR-10 transfer experiments complete. Results saved to cifar10_transfer_results.csv")
