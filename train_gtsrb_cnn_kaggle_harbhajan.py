"""
Train a CNN for German Traffic Sign Recognition on Kaggle.

This version supports both common GTSRB Kaggle layouts:
1) CSV/image layout:
   /kaggle/input/.../Train.csv
   /kaggle/input/.../Test.csv
   /kaggle/input/.../Train/<class>/<image>.png
   /kaggle/input/.../Test/<image>.png

2) Pickle layout used by harbhajansingh21/german-traffic-sign-dataset:
   /kaggle/input/.../train.p
   /kaggle/input/.../valid.p
   /kaggle/input/.../test.p
   /kaggle/input/.../signnames.csv

Run in Kaggle:
    !python /kaggle/working/train_gtsrb_cnn_kaggle_harbhajan.py

Outputs are written to:
    /kaggle/working/outputs_gtsrb_cnn
"""

from __future__ import annotations

import os

# Reduces normal TensorFlow logs. Kaggle may still print early CUDA factory warnings;
# those are usually harmless and do not mean training failed.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = (32, 32)
BATCH_SIZE = 128
EPOCHS = 25
OUTPUT_DIR = Path("/kaggle/working/outputs_gtsrb_cnn")

np.random.seed(SEED)
tf.random.set_seed(SEED)


def configure_gpu() -> None:
    """Enable memory growth where possible so TensorFlow does not reserve all GPU RAM."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs visible to TensorFlow: {len(gpus)}")


def list_kaggle_input() -> None:
    """Print a compact view of files under /kaggle/input for debugging dataset paths."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        print("/kaggle/input does not exist. This is expected outside Kaggle.")
        return

    print("\nVisible Kaggle input files/directories:")
    shown = 0
    for path in sorted(input_root.rglob("*")):
        rel = path.relative_to(input_root)
        print(f"  {rel}")
        shown += 1
        if shown >= 80:
            print("  ... truncated after 80 entries")
            break
    print()


def find_dataset_root() -> Path:
    """Find the directory that contains either GTSRB CSV files or train.p/valid.p/test.p."""
    roots_to_check = [
        Path("/kaggle/input/german-traffic-sign-dataset"),
        Path("/kaggle/input/gtsrb-german-traffic-sign"),
        Path("/kaggle/input"),
        Path.cwd(),
    ]

    for base in roots_to_check:
        if not base.exists():
            continue

        # Direct match.
        if (base / "Train.csv").exists() and (base / "Test.csv").exists():
            return base.resolve()
        if (base / "train.p").exists() and (base / "test.p").exists():
            return base.resolve()

        # Nested match.
        for candidate in base.rglob("*"):
            if not candidate.is_dir():
                continue
            if (candidate / "Train.csv").exists() and (candidate / "Test.csv").exists():
                return candidate.resolve()
            if (candidate / "train.p").exists() and (candidate / "test.p").exists():
                return candidate.resolve()

    list_kaggle_input()
    raise FileNotFoundError(
        "Could not find GTSRB dataset. Expected either Train.csv/Test.csv or "
        "train.p/valid.p/test.p somewhere under /kaggle/input. "
        "Run `!find /kaggle/input -maxdepth 4 -type f | sort | head -100` "
        "inside the notebook and check the actual mount path."
    )


def load_pickle_file(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_sign_names(root: Path) -> Optional[pd.DataFrame]:
    candidates = list(root.rglob("signnames.csv")) + list(root.rglob("SignNames.csv"))
    if not candidates:
        return None
    sign_names = pd.read_csv(candidates[0])
    print(f"Loaded class-name file: {candidates[0]}")
    return sign_names


def load_from_pickle_layout(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Load GTSRB from train.p/valid.p/test.p files."""
    train_path = root / "train.p"
    valid_path = root / "valid.p"
    test_path = root / "test.p"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Pickle layout requires at least train.p and test.p")

    train_data = load_pickle_file(train_path)
    test_data = load_pickle_file(test_path)

    x_train = train_data["features"]
    y_train = train_data["labels"]
    x_test = test_data["features"]
    y_test = test_data["labels"]

    # If validation set exists, merge it into training for final training. We still create
    # a validation split during model.fit, which keeps the code simple and reproducible.
    if valid_path.exists():
        valid_data = load_pickle_file(valid_path)
        x_train = np.concatenate([x_train, valid_data["features"]], axis=0)
        y_train = np.concatenate([y_train, valid_data["labels"]], axis=0)
        print(f"Loaded train.p + valid.p + test.p from: {root}")
    else:
        print(f"Loaded train.p + test.p from: {root}")

    sign_names = load_sign_names(root)
    return x_train, y_train, x_test, y_test, sign_names


def load_from_csv_layout(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Load GTSRB from Train.csv/Test.csv image paths."""
    from PIL import Image

    train_csv = pd.read_csv(root / "Train.csv")
    test_csv = pd.read_csv(root / "Test.csv")

    def read_images(df: pd.DataFrame, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        images: List[np.ndarray] = []
        labels: List[int] = []
        for _, row in df.iterrows():
            rel_path = str(row["Path"])
            image_path = root / rel_path
            if not image_path.exists():
                # Some Kaggle variants store only the filename in Path.
                alt = root / split_name / rel_path
                image_path = alt if alt.exists() else image_path
            img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(int(row["ClassId"]))
        return np.stack(images), np.asarray(labels, dtype=np.int64)

    print(f"Loaded CSV layout from: {root}")
    x_train, y_train = read_images(train_csv, "Train")
    x_test, y_test = read_images(test_csv, "Test")
    sign_names = load_sign_names(root)
    return x_train, y_train, x_test, y_test, sign_names


def load_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    if (root / "train.p").exists():
        return load_from_pickle_layout(root)
    if (root / "Train.csv").exists():
        return load_from_csv_layout(root)
    raise ValueError(f"Unsupported dataset layout at {root}")


def preprocess_images(x: np.ndarray) -> np.ndarray:
    """Convert images to float32 in [0, 1], resizing only if needed."""
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected images with shape (N, H, W, 3), got {x.shape}")

    # Most GTSRB pickle files are already 32x32. Resize only if necessary.
    if tuple(x.shape[1:3]) != IMG_SIZE:
        x = tf.image.resize(x, IMG_SIZE).numpy()

    x = x.astype("float32") / 255.0
    return x


def make_model(num_classes: int) -> keras.Model:
    """Compact CNN suitable for GTSRB classification."""
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = layers.RandomRotation(0.04, seed=SEED)(inputs)
    x = layers.RandomTranslation(0.05, 0.05, seed=SEED)(x)
    x = layers.RandomZoom(0.08, seed=SEED)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="gtsrb_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_class_distribution(y_train: np.ndarray, output_path: Path) -> None:
    counts = pd.Series(y_train).value_counts().sort_index()
    plt.figure(figsize=(12, 4))
    counts.plot(kind="bar")
    plt.title("GTSRB training class distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_training_curves(history: keras.callbacks.History, output_path: Path) -> None:
    hist = pd.DataFrame(history.history)
    plt.figure(figsize=(9, 5))
    plt.plot(hist["accuracy"], label="train accuracy")
    plt.plot(hist["val_accuracy"], label="validation accuracy")
    plt.plot(hist["loss"], label="train loss")
    plt.plot(hist["val_loss"], label="validation loss")
    plt.title("Training and validation curves")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix on test set")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_class_mapping(sign_names: Optional[pd.DataFrame], output_path: Path) -> None:
    if sign_names is None:
        return
    sign_names.to_csv(output_path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_gpu()

    root = find_dataset_root()
    print(f"Dataset root: {root}")

    x_train, y_train, x_test, y_test, sign_names = load_dataset(root)
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    num_classes = int(max(y_train.max(), y_test.max())) + 1
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")
    print(f"Detected classes: {num_classes}")

    plot_class_distribution(y_train, OUTPUT_DIR / "class_distribution.png")
    save_class_mapping(sign_names, OUTPUT_DIR / "class_names.csv")

    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_values)}

    model = make_model(num_classes=num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "best_gtsrb_cnn.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_proba = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_proba, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        output_dict=True,
        zero_division=0,
    )

    metrics_df = pd.DataFrame(
        [{"test_loss": float(test_loss), "test_accuracy": float(test_accuracy), "num_classes": num_classes}]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix.png")

    model.save(OUTPUT_DIR / "final_gtsrb_cnn.keras")

    print("\nSaved outputs:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
