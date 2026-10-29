"""Train a violence / non-violence frame classifier on the Real Life Violence Situations dataset.

Transfer learning: a frozen MobileNetV2 backbone (ImageNet weights) with a
sigmoid head, trained on frames sampled from each video.

The train/test split is made per VIDEO, not per frame, so frames from one
clip never appear in both sets (a frame-level split leaks near-duplicate
frames and inflates test accuracy). Augmentation is applied to training
frames only.

Usage:
    python train.py --data "Real Life Violence Dataset" --videos-per-class 350
"""

import argparse
import os
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input
from tqdm import tqdm

SEED = 73
IMG_SIZE = 128
CLASSES = ["NonViolence", "Violence"]
FRAME_STEP = 7  # keep every 7th frame to avoid near-duplicates

AUGMENT = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Multiply((1.0, 1.3)),
    iaa.Affine(scale=(1.0, 1.3), rotate=(-25, 25)),
])


def video_to_frames(path: str, augment: bool) -> list[np.ndarray]:
    """Sample every FRAME_STEP-th frame, optionally augment, return RGB IMG_SIZE x IMG_SIZE arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok, image = cap.read()
        if not ok:
            break
        if idx % FRAME_STEP == 0:
            if augment:
                image = AUGMENT(image=image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)))
    cap.release()
    return frames


def build_dataset(videos: list[tuple[str, int]], augment: bool):
    X, y = [], []
    for path, label in tqdm(videos):
        frames = video_to_frames(path, augment)
        X.extend(frames)
        y.extend([label] * len(frames))
    return np.array(X, dtype="float32") / 255.0, np.array(y)


def build_model() -> Model:
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(pooling="avg", include_top=False, input_tensor=inputs)
    base.trainable = False
    outputs = Dense(1, activation="sigmoid")(base.output)
    model = Model(inputs, outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def lr_schedule(epoch, start_lr=1e-5, max_lr=5e-5, min_lr=1e-5, rampup=5, decay=0.8):
    """Linear warm-up for `rampup` epochs, then exponential decay."""
    if epoch < rampup:
        return (max_lr - start_lr) / rampup * epoch + start_lr
    return (max_lr - min_lr) * decay ** (epoch - rampup) + min_lr


class StopAtAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs or {}).get("accuracy", 0) >= 0.999:
            print("\nTraining accuracy limit reached, stopping.")
            self.model.stop_training = True


def plot_history(history, out_dir: Path) -> None:
    for metric in ("loss", "accuracy"):
        plt.figure()
        plt.plot(history.history[metric], label="training")
        plt.plot(history.history[f"val_{metric}"], label="validation")
        plt.title(f"Training and validation {metric}")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(out_dir / f"{metric}.png", dpi=120)
        plt.close()


def main(args) -> None:
    tf.random.set_seed(SEED)
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    videos = []
    for label, cls in enumerate(CLASSES):
        folder = Path(args.data) / cls
        files = sorted(os.listdir(folder))[: args.videos_per_class]
        videos += [(str(folder / f), label) for f in files]

    train_videos, test_videos = train_test_split(
        videos, test_size=0.3, random_state=SEED, stratify=[label for _, label in videos]
    )
    print(f"{len(train_videos)} training videos, {len(test_videos)} test videos")

    X_train, y_train = build_dataset(train_videos, augment=True)
    X_test, y_test = build_dataset(test_videos, augment=False)
    print(f"{len(X_train)} training frames, {len(X_test)} test frames")

    model = build_model()
    weights = str(out_dir / "best.weights.h5")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            StopAtAccuracy(),
            LearningRateScheduler(lr_schedule),
            ModelCheckpoint(weights, save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True),
            EarlyStopping(patience=3, monitor="val_loss", mode="min", restore_best_weights=True, min_delta=0.00075),
            ReduceLROnPlateau(patience=2, mode="min"),
        ],
    )
    model.load_weights(weights)
    plot_history(history, out_dir)

    preds = (model.predict(X_test) > 0.5).astype(int).ravel()
    cm = confusion_matrix(y_test, preds)
    print(f"Correct: {np.trace(cm)}, wrong: {cm.sum() - np.trace(cm)}")
    report = classification_report(y_test, preds, target_names=CLASSES)
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    model.save(out_dir / "model.h5")
    print(f"Saved model and plots to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the violence detection model.")
    parser.add_argument("--data", required=True, help="dataset folder containing Violence/ and NonViolence/")
    parser.add_argument("--videos-per-class", type=int, default=350, help="cap per class to fit in memory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="results")
    main(parser.parse_args())
