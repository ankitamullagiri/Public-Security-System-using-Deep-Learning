# Public Security System: Violence Detection in CCTV-style Video

A deep-learning system that watches video and flags violent activity in real time, built as a step towards automated alerts for public-space surveillance. It pairs with the [Human Detection](https://github.com/ankitamullagiri/Human-detection) project.

## Problem

Security teams can't watch every camera feed. A model that flags likely violence lets a human operator focus on the few feeds that matter.

## Data

[Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) (Kaggle): 2,000 short clips, 1,000 violent and 1,000 non-violent, from street fights, sports and everyday scenes. Because of memory limits on Colab, 350 videos per class were used.

## Approach

| Step | Details |
|---|---|
| Frame sampling | Every 7th frame per video, resized to 128×128 RGB |
| Split | **70/30 by video**, stratified, so frames from one clip never appear in both train and test |
| Augmentation (train only) | Random flip, brightness, zoom and rotation via `imgaug` |
| Model | MobileNetV2 (ImageNet weights, frozen) + a sigmoid output layer |
| Training | Adam, binary cross-entropy, LR warm-up then exponential decay, early stopping on validation loss |
| Inference | Per-frame probability, smoothed with a rolling average over the last 128 frames to stop the label flickering |

## Run it

```bash
pip install -r requirements.txt

# 1. Train (expects the dataset folder with Violence/ and NonViolence/ inside)
python train.py --data "Real Life Violence Dataset" --videos-per-class 350

# 2. Predict on a video
python predict.py --model results/model.h5 --video clip.mp4 --show
```

Training saves `model.h5`, loss/accuracy curves, the confusion matrix and the classification report to `results/`.

Original Colab notebooks: [training](https://colab.research.google.com/drive/1WB9-N41SFA3PKZ6UpxTdT69guszquK8Y?usp=sharing) · [prediction](https://colab.research.google.com/drive/1c0C8a55qAZ42S7Fn2aDnuQ9oXONTy1gm?usp=sharing)

## Limitations and next steps

- The model classifies single frames, so it can't see motion directly. A punch and a high-five can look alike in one frame.
- **Next steps:** a temporal model (CNN + LSTM, or a 3D CNN like I3D) on frame sequences; fine-tune the top MobileNetV2 layers; and combine with the person detector to only score frames containing people.

## Acknowledgements

The training pipeline was adapted from a public Kaggle notebook for this dataset, then restructured into scripts with a leak-free per-video split.

## Tech stack

Python · TensorFlow/Keras · MobileNetV2 · OpenCV · imgaug · scikit-learn
