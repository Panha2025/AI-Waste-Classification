"""Train direct two-output classifiers on provisional source-derived binary labels."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd

from waste_ai.binary_data import BINARY_LABELS, OUTPUT as MANIFEST
from waste_ai.config import ARTIFACTS, IMAGE_SIZE, METRICS, MODELS, SEED, ensure_dirs
from waste_ai.hardware import hardware_report
from waste_ai.metrics import summarize_predictions
from waste_ai.training import read_image_pil, training_class_weights, augmentation, plot_history

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(BINARY_LABELS)}


def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError("Run `python -m waste_ai.binary_data` first")
    frame = pd.read_csv(MANIFEST)
    required = {"path", "binary_label", "capture_group", "split", "sha256", "label_origin"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Binary manifest missing columns: {sorted(required - set(frame.columns))}")
    return frame


def make_binary_dataset(frame, batch_size=32, training=False, seed=SEED):
    import tensorflow as tf
    paths = frame.path.astype(str).to_numpy()
    labels = frame.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(frame), 2000), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.numpy_function(lambda value: read_image_pil(value.item() if hasattr(value, "item") else value),
                                  [path], tf.float32)
        image.set_shape((IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        return image, label

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_binary_model(name, pretrained=True):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="waste_rgb_image")
    x = augmentation()(inputs)
    backbone = None
    if name == "mobilenetv2":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
        # Reuse the locally saved six-class experiment's frozen ImageNet backbone
        # when available. Only feature weights are copied; its six-way head/predictions
        # are never used by this binary model.
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=None)
        local_source = MODELS / "mobilenetv2.keras"
        if pretrained and local_source.is_file():
            source_model = tf.keras.models.load_model(local_source, compile=False)
            source_backbone = next((layer for layer in source_model.layers
                                    if isinstance(layer, tf.keras.Model) and
                                    layer.name.startswith("mobilenetv2_1.00_224")), None)
            if source_backbone is None:
                raise ValueError("Could not locate the saved pretrained MobileNetV2 backbone")
            backbone.set_weights(source_backbone.get_weights())
            del source_backbone, source_model
        elif pretrained:
            backbone = tf.keras.applications.MobileNetV2(
                input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
        backbone.trainable = not pretrained
        x = backbone(x, training=False) if pretrained else backbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif name == "custom_cnn":
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
    else:
        raise ValueError(f"Unsupported binary model: {name}")
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="binary_waste_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name=f"binary_{name}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model


def train_one(name="mobilenetv2", epochs=8, batch_size=32, patience=2, weights="imagenet"):
    import tensorflow as tf
    ensure_dirs()
    frame = load_manifest()
    train = frame[frame.split == "train"].copy()
    val = frame[frame.split == "val"].copy()
    if set(train.binary_label) != set(BINARY_LABELS) or set(val.binary_label) != set(BINARY_LABELS):
        raise ValueError("Train and validation splits must each contain both binary classes")
    tf.keras.utils.set_random_seed(SEED)
    model = build_binary_model(name, pretrained=(weights == "imagenet"))
    weights_by_class = training_class_weights(train.binary_label.map(LABEL_TO_INDEX).to_numpy())
    train_ds = make_binary_dataset(train, batch_size, training=True)
    val_ds = make_binary_dataset(val, batch_size)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    fitted = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
                       class_weight=weights_by_class, verbose=1, shuffle=False)
    duration = round(time.perf_counter() - start, 2)
    history = {key: [float(x) for x in values] for key, values in fitted.history.items()}
    val_probs = model.predict(val_ds, verbose=0)
    val_truth = val.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=int)
    val_metrics = summarize_predictions(val_truth, val_probs)
    model_path = MODELS / f"binary_{name}.keras"
    model.save(model_path)
    record = {
        "model": name, "task": "direct binary softmax classification",
        "model_path": str(model_path), "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": SEED, "epochs_requested": epochs, "epochs_completed": len(history.get("loss", [])),
            "batch_size": batch_size, "patience": patience, "optimizer": "Adam",
            "learning_rate": 1e-3,
            "pretrained_weights": ("local frozen ImageNet MobileNetV2 backbone copied from six-class model"
                                   if name == "mobilenetv2" and weights == "imagenet" and
                                   (MODELS / "mobilenetv2.keras").is_file() else weights),
            "class_weight": {BINARY_LABELS[int(k)]: float(v) for k, v in weights_by_class.items()},
            "image_size": IMAGE_SIZE, "class_order": BINARY_LABELS,
            "train_images": int(len(train)), "validation_images": int(len(val)),
            "label_origin": "source-folder mapped, provisional, not human-verified",
        },
        "training_seconds": duration, "hardware": hardware_report(),
        "validation_metrics": val_metrics, "history": history,
    }
    (METRICS / f"binary_{name}_train.json").write_text(json.dumps(record, indent=2), encoding="utf8")
    plot_history(f"binary_{name}", history)
    print(f"Saved {model_path}; validation macro F1={val_metrics['macro_f1']:.4f}; "
          f"training={duration:.1f}s")
    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a direct binary waste classifier")
    parser.add_argument("--model", choices=("mobilenetv2", "custom_cnn"), default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        parser.error("epochs and batch-size must be positive, patience non-negative")
    train_one(args.model, args.epochs, args.batch_size, args.patience, args.weights)


if __name__ == "__main__":
    main()
