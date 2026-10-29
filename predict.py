"""Flag violence in a video with the trained MobileNetV2 classifier.

Each frame is classified, and the label shown is based on a rolling average
of the last N frame predictions, which smooths out single-frame flicker.

Usage:
    python predict.py --model results/model.h5 --video clip.mp4
    python predict.py --model results/model.h5 --video clip.mp4 --show
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128


def predict_video(model_path: str, video_path: str, output_path: str, window: int, threshold: float, show: bool) -> float:
    print("Loading model...")
    model = load_model(model_path)
    recent = deque(maxlen=window)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    violent_frames = total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        x = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        recent.append(float(model.predict(x[np.newaxis], verbose=0)[0][0]))
        score = float(np.mean(recent))
        violent = score > threshold

        total += 1
        violent_frames += violent
        color = (0, 0, 255) if violent else (0, 255, 0)  # red / green (BGR)
        cv2.putText(frame, f"Violence: {violent} ({score:.2f})", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)
        writer.write(frame)

        if show:
            cv2.imshow("Violence detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    share = violent_frames / max(total, 1)
    print(f"{violent_frames}/{total} frames flagged ({share:.0%}) -> {output_path}")
    return share


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect violence in a video.")
    parser.add_argument("--model", required=True, help="trained .h5 model from train.py")
    parser.add_argument("--video", required=True, help="input video file")
    parser.add_argument("--output", default="output/annotated.mp4")
    parser.add_argument("--window", type=int, default=128, help="frames in the rolling average")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show", action="store_true", help="display frames while processing")
    args = parser.parse_args()
    predict_video(args.model, args.video, args.output, args.window, args.threshold, args.show)
