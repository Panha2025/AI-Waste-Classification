"""Train and compare a baseline CNN, MobileNetV2, and EfficientNetB0.

Final test sets are NOT used during training/model selection. Evaluate them
only after model selection using `python -m waste_ai.evaluate`.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.utils.class_weight import compute_class_weight

from waste_ai.config import (IMAGE_SIZE, LABELS, CLASS_TO_INDEX, PROCESSED, MODELS,
                             METRICS, SEED, ensure_dirs)
from waste_ai.hardware import hardware_report
from waste_ai.metrics import summarize_predictions

SUPPORTED = ("custom_cnn", "mobilenetv2", "efficientnetb0")


def load_manifest():
    path = PROCESSED / "split_manifest.csv"
    if not path.exists():
        raise FileNotFoundError("Run 'python -m waste_ai.data' before training")
    df = pd.read_csv(path)
    required = {"path", "label", "source", "split", "capture_group"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest missing: {sorted(required - set(df.columns))}")
    return df


def read_image_pil(path, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1]):
    """Aspect-preserving RGB letterbox; retains pixel range [0,255]."""
    if isinstance(path, (bytes, np.bytes_)):
        path = path.decode("utf8")
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image = ImageOps.pad(image, (width, height), method=Image.Resampling.BILINEAR,
                             color=(0, 0, 0), centering=(0.5, 0.5))
        return np.asarray(image, dtype=np.float32)


def training_class_weights(labels):
    """Balanced weights computed from training labels only; returns Keras class map."""
    values = np.asarray(labels, dtype=int)
    classes = np.unique(values)
    if len(classes) < 2:
        raise ValueError("Class weighting requires both classes in the training set")
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=values)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def make_dataset(frame, batch_size=16, training=False, seed=SEED):
    import tensorflow as tf
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["label"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(frame), 2000), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.numpy_function(lambda s: read_image_pil(s.item() if hasattr(s, 'item') else s), [path], tf.float32)
        image.set_shape((IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        return image, label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augmentation():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.12),
    ], name="training_only_augmentation")


def build_model(model_name, weights="imagenet"):
    import tensorflow as tf
    if model_name not in SUPPORTED:
        raise ValueError(f"Unknown model: {model_name}")
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_waste_image")
    x = augmentation()(inputs)
    base = None
    if model_name == "custom_cnn":
        x = tf.keras.layers.Rescaling(1. / 255.)(x)
        for size in (32, 64, 128):
            x = tf.keras.layers.Conv2D(size, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
    elif model_name == "mobilenetv2":
        x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(x)
        base = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3),
                    include_top=False, weights=None if weights == "none" else "imagenet")
        base.trainable = (weights == "none")
        # For pretrained transfer learning, force inference-mode batch normalization.
        # With weights='none', actually train the backbone from scratch.
        x = base(x) if weights == "none" else base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        # EfficientNetB0 includes its own pixel scaling; feed raw [0,255].
        base = tf.keras.applications.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3),
                    include_top=False, weights=None if weights == "none" else "imagenet")
        base.trainable = (weights == "none")
        # For pretrained transfer learning, force inference-mode batch normalization.
        # With weights='none', actually train the backbone from scratch.
        x = base(x) if weights == "none" else base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(len(LABELS), activation="softmax", name="waste_category_probabilities")(x)
    return tf.keras.Model(inputs, output, name=model_name), base


def compile_model(model, learning_rate):
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def train_one(name, frame, args):
    import tensorflow as tf
    print(f"\nTraining {name} ...")
    tf.keras.utils.set_random_seed(args.seed)
    training = frame[frame.split == "train"]
    val = frame[frame.split == "val"]
    if len(training) == 0 or len(val) == 0:
        raise ValueError("Train/validation split is empty. Re-run dataset preparation.")
    train_ds = make_dataset(training, args.batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(val, args.batch_size)
    model, backbone = build_model(name, weights=args.weights)
    compile_model(model, learning_rate=args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    class_weight = (training_class_weights(training["label"].map(CLASS_TO_INDEX).to_numpy())
                    if args.class_weight == "balanced" else None)
    start = time.perf_counter()
    history_1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                          callbacks=callbacks, verbose=1, class_weight=class_weight, shuffle=False)
    history = {k: [float(x) for x in v] for k, v in history_1.history.items()}
    if backbone is not None and args.fine_tune_epochs:
        # Fine-tune only the last ~20 non-BN layers, avoid BN drift on small data.
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, learning_rate=args.fine_tune_learning_rate)
        history_2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                              callbacks=callbacks, verbose=1, class_weight=class_weight, shuffle=False)
        for key, values in history_2.history.items():
            history.setdefault(key, []).extend(float(x) for x in values)
    seconds = round(time.perf_counter() - start, 2)
    probabilities = model.predict(val_ds, verbose=0)
    actual = val["label"].map(CLASS_TO_INDEX).to_numpy(dtype=int)
    val_metrics = summarize_predictions(actual, probabilities)
    model_file = MODELS / f"{name}.keras"
    model.save(model_file)
    record = {
        "model": name, "model_path": str(model_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": args.seed, "epochs_initial_requested": args.epochs,
            "epochs_finetune_requested": args.fine_tune_epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "fine_tune_learning_rate": args.fine_tune_learning_rate,
            "patience": args.patience, "weights": args.weights,
            "class_weight": args.class_weight,
            "image_size": IMAGE_SIZE, "class_order": LABELS,
            "train_images": int(len(training)), "validation_images": int(len(val)),
        },
        "training_seconds": seconds, "hardware": hardware_report(),
        "validation_metrics": val_metrics, "history": history,
    }
    (METRICS / f"{name}_train.json").write_text(json.dumps(record, indent=2), encoding="utf8")
    plot_history(name, history)
    print(f"Saved {model_file} | Validation macro F1 = {val_metrics['macro_f1']:.3f}")
    return record


def plot_history(name, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        for curve in (key, f"val_{key}"):
            if curve in history:
                ax.plot(history[curve], label=curve)
        ax.set_title(f"{name} — {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(METRICS / f"{name}_history.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Train CNN/MobileNetV2/EfficientNetB0 on real images")
    p.add_argument("--model", choices=("all", *SUPPORTED), default="all")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--fine-tune-epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--fine-tune-learning-rate", type=float, default=0.00001)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    p.add_argument("--class-weight", choices=("none", "balanced"), default="none",
                   help="Apply balanced class weights computed from training labels only")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.fine_tune_epochs < 0:
        p.error("Epochs/batch size must be positive; fine-tune epochs must be non-negative")
    ensure_dirs()
    frame = load_manifest()
    print("Training uses only train/val. Test sets are reserved for evaluation.")
    names = SUPPORTED if args.model == "all" else (args.model,)
    for name in names:
        train_one(name, frame, args)
    print("\nNext: python -m waste_ai.evaluate --all")


if __name__ == "__main__":
    main()
