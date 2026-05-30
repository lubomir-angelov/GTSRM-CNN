"""
Regularized CNN for German Traffic Sign Recognition on Kaggle.

This version is intended to reduce the train/validation gap observed in the
first experiment. It supports both common GTSRB Kaggle layouts:

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
    !python /kaggle/working/train_gtsrb_cnn_kaggle_regularized.py

Outputs are written to:
    /kaggle/working/outputs_gtsrb_cnn_regularized

Main changes compared with the first version:
- use the official valid.p validation set instead of merging it into training;
- avoid Keras validation_split on ordered arrays;
- use a smaller regularized CNN with L2 weight decay and SpatialDropout;
- use GlobalAveragePooling2D instead of a large Flatten + Dense head;
- disable class weights by default, because they can overcompensate class
  imbalance and make validation loss unstable;
- add stronger but still realistic image augmentation;
- save training_history.csv and metrics.csv for direct coursework reporting.
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

SEED = 42
IMG_SIZE = (32, 32)
BATCH_SIZE = 128
EPOCHS = 40
VALIDATION_FRACTION = 0.15
USE_CLASS_WEIGHTS = False
OUTPUT_DIR = Path("/kaggle/working/outputs_gtsrb_cnn_regularized")

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

        if (base / "Train.csv").exists() and (base / "Test.csv").exists():
            return base.resolve()
        if (base / "train.p").exists() and (base / "test.p").exists():
            return base.resolve()

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


def load_from_pickle_layout(
    root: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Load GTSRB from train.p/valid.p/test.p files.

    Important: unlike the first script, this function does NOT merge valid.p into
    train.p. The official validation split is kept separate so validation metrics
    are meaningful and not affected by Keras validation_split on ordered arrays.
    """
    train_path = root / "train.p"
    valid_path = root / "valid.p"
    test_path = root / "test.p"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Pickle layout requires at least train.p and test.p")

    train_data = load_pickle_file(train_path)
    test_data = load_pickle_file(test_path)

    x_train = train_data["features"]
    y_train = train_data["labels"]
    x_valid: Optional[np.ndarray] = None
    y_valid: Optional[np.ndarray] = None

    if valid_path.exists():
        valid_data = load_pickle_file(valid_path)
        x_valid = valid_data["features"]
        y_valid = valid_data["labels"]
        print(f"Loaded train.p + valid.p + test.p from: {root}")
    else:
        print(f"Loaded train.p + test.p from: {root}; will create a stratified validation split")

    x_test = test_data["features"]
    y_test = test_data["labels"]
    sign_names = load_sign_names(root)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, sign_names


def load_from_csv_layout(
    root: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
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
    return x_train, y_train, None, None, x_test, y_test, sign_names


def load_dataset(
    root: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
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

    if tuple(x.shape[1:3]) != IMG_SIZE:
        x = tf.image.resize(x, IMG_SIZE).numpy()

    return x.astype("float32") / 255.0


def make_validation_split_if_needed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use official validation data where available; otherwise create stratified validation data."""
    if x_valid is not None and y_valid is not None:
        return x_train, y_train, x_valid, y_valid

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=y_train,
        shuffle=True,
    )
    print(f"Created stratified validation split with fraction {VALIDATION_FRACTION}")
    return x_tr, y_tr, x_val, y_val


def make_regularized_model(num_classes: int, x_train_for_norm: np.ndarray) -> keras.Model:
    """Smaller CNN with stronger regularization for better generalization."""
    reg = regularizers.l2(1e-4)

    normalization = layers.Normalization(axis=-1, name="image_normalization")
    normalization.adapt(x_train_for_norm)

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Data augmentation is active during training and inactive during validation/test.
    x = layers.RandomContrast(0.15, seed=SEED)(inputs)
    x = layers.RandomRotation(0.06, seed=SEED)(x)
    x = layers.RandomTranslation(0.08, 0.08, seed=SEED)(x)
    x = layers.RandomZoom(0.10, seed=SEED)(x)
    x = normalization(x)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.10)(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.15)(x)

    x = layers.Conv2D(96, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(96, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SpatialDropout2D(0.20)(x)

    # GlobalAveragePooling drastically reduces the classifier parameter count compared
    # with Flatten, which is a common source of overfitting on small images.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="gtsrb_cnn_regularized")

    try:
        optimizer = keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4)
    except AttributeError:
        optimizer = keras.optimizers.Adam(learning_rate=5e-4)

    model.compile(
        optimizer=optimizer,
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
    if sign_names is not None:
        sign_names.to_csv(output_path, index=False)


def compute_class_weights_if_enabled(y_train: np.ndarray, num_classes: int) -> Optional[Dict[int, float]]:
    if not USE_CLASS_WEIGHTS:
        return None
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    return {i: float(w) for i, w in enumerate(weights)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_gpu()

    root = find_dataset_root()
    print(f"Dataset root: {root}")

    x_train, y_train, x_valid, y_valid, x_test, y_test, sign_names = load_dataset(root)
    x_train = preprocess_images(x_train)
    x_valid = preprocess_images(x_valid) if x_valid is not None else None
    x_test = preprocess_images(x_test)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_valid = np.asarray(y_valid, dtype=np.int64) if y_valid is not None else None
    y_test = np.asarray(y_test, dtype=np.int64)

    x_train, y_train, x_valid, y_valid = make_validation_split_if_needed(x_train, y_train, x_valid, y_valid)

    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max())) + 1
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_valid: {x_valid.shape}, y_valid: {y_valid.shape}")
    print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")
    print(f"Detected classes: {num_classes}")
    print(f"Use class weights: {USE_CLASS_WEIGHTS}")

    plot_class_distribution(y_train, OUTPUT_DIR / "class_distribution.png")
    save_class_mapping(sign_names, OUTPUT_DIR / "class_names.csv")

    class_weights = compute_class_weights_if_enabled(y_train, num_classes)

    model = make_regularized_model(num_classes=num_classes, x_train_for_norm=x_train)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "best_gtsrb_cnn_regularized.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    val_loss, val_accuracy = model.evaluate(x_valid, y_valid, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
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

    final_train_acc = float(hist_df["accuracy"].iloc[-1])
    best_val_acc = float(hist_df["val_accuracy"].max())
    metrics_df = pd.DataFrame(
        [
            {
                "validation_loss": float(val_loss),
                "validation_accuracy": float(val_accuracy),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "final_train_accuracy": final_train_acc,
                "best_validation_accuracy": best_val_acc,
                "train_validation_accuracy_gap": final_train_acc - best_val_acc,
                "num_classes": num_classes,
                "train_images": int(len(x_train)),
                "validation_images": int(len(x_valid)),
                "test_images": int(len(x_test)),
                "use_class_weights": USE_CLASS_WEIGHTS,
            }
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix.png")

    model.save(OUTPUT_DIR / "final_gtsrb_cnn_regularized.keras")

    with (OUTPUT_DIR / "experiment_notes.txt").open("w", encoding="utf-8") as f:
        f.write(
            "Regularized GTSRB CNN experiment\n"
            "=================================\n"
            "This run keeps valid.p as an explicit validation set when available. "
            "It uses L2 regularization, SpatialDropout2D, GlobalAveragePooling2D, "
            "AdamW weight decay, stronger data augmentation, and class weights are "
            f"set to {USE_CLASS_WEIGHTS}.\n"
        )

    print("\nSaved outputs:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
