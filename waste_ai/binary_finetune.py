"""Fine-tune the upper MobileNetV2 layers with extra training-only minority weighting."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd

from waste_ai.binary_data import BINARY_LABELS, OUTPUT as MANIFEST
from waste_ai.binary_training import LABEL_TO_INDEX, make_binary_dataset
from waste_ai.config import METRICS, MODELS, SEED
from waste_ai.hardware import hardware_report
from waste_ai.metrics import summarize_predictions
from waste_ai.training import plot_history, training_class_weights

BASELINE = MODELS / "binary_mobilenetv2.keras"
OUTPUT = MODELS / "binary_mobilenetv2_finetuned.keras"


def train_finetuned(epochs=6, batch_size=32, patience=2, top_layers=20,
                    non_recyclable_weight_multiplier=2.0):
    import tensorflow as tf
    if not BASELINE.is_file():
        raise FileNotFoundError(f"Binary MobileNetV2 baseline not found: {BASELINE}")
    if top_layers < 1 or non_recyclable_weight_multiplier < 1:
        raise ValueError("top_layers and minority weight multiplier must be positive (multiplier >= 1)")
    frame = pd.read_csv(MANIFEST)
    train = frame[frame.split == "train"].copy()
    val = frame[frame.split == "val"].copy()
    class_weights = training_class_weights(train.binary_label.map(LABEL_TO_INDEX).to_numpy())
    class_weights[LABEL_TO_INDEX["non_recyclable"]] *= non_recyclable_weight_multiplier
    train_ds = make_binary_dataset(train, batch_size, training=True)
    val_ds = make_binary_dataset(val, batch_size)

    model = tf.keras.models.load_model(BASELINE, compile=False)
    backbone = next((layer for layer in model.layers
                     if isinstance(layer, tf.keras.Model) and
                     layer.name.startswith("mobilenetv2_1.00_224")), None)
    if backbone is None:
        raise ValueError("Could not find the nested MobileNetV2 backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-top_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    fit = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
                    class_weight=class_weights, verbose=1, shuffle=False)
    duration = round(time.perf_counter() - start, 2)
    history = {key: [float(value) for value in values] for key, values in fit.history.items()}
    val_probs = np.asarray(model.predict(val_ds, verbose=0), dtype=float)
    val_truth = val.binary_label.map(LABEL_TO_INDEX).to_numpy(dtype=int)
    val_metrics = summarize_predictions(val_truth, val_probs)
    model.save(OUTPUT)
    record = {
        "model": "mobilenetv2_finetuned", "task": "direct binary softmax classification",
        "model_path": str(OUTPUT), "initialized_from": str(BASELINE),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": SEED, "epochs_requested": epochs, "epochs_completed": len(history.get("loss", [])),
            "batch_size": batch_size, "patience": patience, "optimizer": "Adam",
            "fine_tune_learning_rate": 1e-5, "unfrozen_upper_backbone_layers": top_layers,
            "batch_normalization_trainable": False,
            "class_weight": {BINARY_LABELS[int(key)]: float(value) for key, value in class_weights.items()},
            "non_recyclable_weight_multiplier": non_recyclable_weight_multiplier,
            "image_size": [224, 224], "class_order": list(BINARY_LABELS),
            "train_images": int(len(train)), "validation_images": int(len(val)),
            "label_origin": "source-folder mapped, provisional, not human-verified",
        },
        "training_seconds": duration, "hardware": hardware_report(),
        "validation_metrics_argmax_threshold_0_5": val_metrics, "history": history,
    }
    (METRICS / "binary_mobilenetv2_finetuned_train.json").write_text(
        json.dumps(record, indent=2), encoding="utf8")
    plot_history("binary_mobilenetv2_finetuned", history)
    print(f"Saved {OUTPUT}; validation argmax macro F1={val_metrics['macro_f1']:.4f}; "
          f"training={duration:.1f}s")
    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune upper layers of binary MobileNetV2")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--top-layers", type=int, default=20)
    parser.add_argument("--non-recyclable-weight-multiplier", type=float, default=2.0)
    args = parser.parse_args()
    train_finetuned(args.epochs, args.batch_size, args.patience,
                    args.top_layers, args.non_recyclable_weight_multiplier)


if __name__ == "__main__":
    main()
