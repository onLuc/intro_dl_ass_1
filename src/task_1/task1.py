import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse

# --------------------
# Load Fashion-MNIST
# --------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# --------------------
# Model Builders
# --------------------
def build_mlp_model(input_shape):
    """Basic MLP model (Task 1.2.a)"""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model

def build_cnn_model(input_shape):
    """CNN model (Task 1.2.b)"""
    model = keras.Sequential([
        keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model

# --------------------
# Training + Evaluation
# --------------------
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=20):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=0.1,
        batch_size=64,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Plot training history
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.gca().set_ylim(0, 1)
    plt.grid(True)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# --------------------
# Main Function
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP or CNN on Fashion-MNIST")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn"],
        default="mlp",
        help="Model type: 'mlp' or 'cnn'"
    )
    args = parser.parse_args()

    if args.model == "cnn":
        # Add channel dimension for CNN
        x_train_cnn = x_train[..., tf.newaxis]
        x_test_cnn = x_test[..., tf.newaxis]
        model = build_cnn_model((28, 28, 1))
        print("Training CNN model...")
        train_and_evaluate(model, x_train_cnn, y_train, x_test_cnn, y_test)
    else:
        # MLP uses flattened input
        model = build_mlp_model((28, 28))
        print("Training MLP model...")
        train_and_evaluate(model, x_train, y_train, x_test, y_test)
