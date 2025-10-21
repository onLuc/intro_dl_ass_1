import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import itertools
import os
import pandas as pd

# --------------------
# Load Fashion-MNIST
# --------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# --------------------
# Model builders
# --------------------
def build_mlp_model(input_shape, activation="relu", initializer="glorot_uniform",
                    regularizer=None, dropout_rate=None):
    reg = None
    if regularizer == "l1":
        reg = regularizers.l1(0.001)
    elif regularizer == "l2":
        reg = regularizers.l2(0.001)

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


def build_cnn_model(input_shape, activation="relu", initializer="glorot_uniform",
                    regularizer=None, dropout_rate=None):
    reg = None
    if regularizer == "l1":
        reg = regularizers.l1(0.001)
    elif regularizer == "l2":
        reg = regularizers.l2(0.001)

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
                           kernel_initializer=initializer, kernel_regularizer=reg),
    ])
    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(64, activation=activation,
                                 kernel_initializer=initializer, kernel_regularizer=reg))
    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


# --------------------
# Training + evaluation
# --------------------
def train_and_evaluate(model, x_train, y_train, x_test, y_test,
                       optimizer, epochs=10, batch_size=64, tag="experiment",
                       plot_dir="plots/mlp"):
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=0.1,
        batch_size=batch_size,
        verbose=0
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Save plot
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.gca().set_ylim(0, 1)
    plt.grid(True)
    plt.title(f"{tag} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{tag}.png")
    plt.savefig(plot_path)
    plt.close()

    return test_acc, test_loss, plot_path


# --------------------
# Grid search helper
# --------------------
def run_experiments(model_type):
    activations = ["relu", "tanh"]
    initializers = ["he_uniform", "glorot_uniform"]
    optimizers = {
        "adam": keras.optimizers.Adam(learning_rate=0.001),
        "sgd": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    }
    regularizers_list = [None, "l1", "l2"]
    dropout_rates = [None, 0.3]

    results = []
    total = (len(activations) * len(initializers) *
             len(optimizers) * len(regularizers_list) *
             len(dropout_rates))
    i = 1

    for activation, initializer, (opt_name, optimizer), reg, drop in itertools.product(
        activations, initializers, optimizers.items(), regularizers_list, dropout_rates
    ):
        tag = f"{model_type}_{i}_{activation}_{initializer}_{opt_name}_{reg}_drop{drop}"
        print(f"\n[{i}/{total}] Running {tag} ...")

        if model_type == "cnn":
            xtr = x_train[..., tf.newaxis]
            xte = x_test[..., tf.newaxis]
            model = build_cnn_model((28, 28, 1), activation, initializer, reg, drop)
            plot_dir = "plots/cnn"
        else:
            xtr, xte = x_train, x_test
            model = build_mlp_model((28, 28), activation, initializer, reg, drop)
            plot_dir = "plots/mlp"

        acc, loss, plot_path = train_and_evaluate(
            model, xtr, y_train, xte, y_test, optimizer, epochs=25,
            tag=tag, plot_dir=plot_dir
        )
        print(f"→ {model_type.upper()} | Test acc: {acc:.4f}, loss: {loss:.4f}")

        results.append({
            "model": model_type,
            "activation": activation,
            "initializer": initializer,
            "optimizer": opt_name,
            "regularizer": reg,
            "dropout": drop,
            "test_acc": acc,
            "test_loss": loss,
            "plot": plot_path
        })
        i += 1

    df = pd.DataFrame(results)
    out_csv = f"{model_type}_results_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved results to {out_csv}")


# --------------------
# Main (run both)
# --------------------
if __name__ == "__main__":
    print("\n=== Running MLP Experiments ===")
    run_experiments("mlp")

    print("\n=== Running CNN Experiments ===")
    run_experiments("cnn")

    print("\nAll experiments completed successfully!")
