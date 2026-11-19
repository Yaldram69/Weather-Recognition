# Weather Recognition

A PyTorch CNN that classifies a photo into one of four weather or lighting conditions: `cloudy`, `rain`, `shine`, or `sunrise`. The repository includes the training script, a single-image test script, two trained checkpoints, and the labelled image set used for training.

## Features

- Four-class CNN with four convolution blocks (BatchNorm and max-pooling), dropout, and two fully connected layers. Input is resized to 150x150.
- Training script with an 80/10/10 random train/validation/test split, Adam optimizer, step learning-rate decay, and best-model checkpointing on validation loss.
- Single-image prediction that prints the label and shows the image with the label drawn on it.

## Project structure

- `kaka.py` - training script. Reads `./weather_data`, trains for 15 epochs, saves `best_weather_model.pth` (state dict) and `final_weather_model.pth` (full model).
- `Test.py` - loads `final_weather_model.pth` and predicts the weather for one image.
- `best_weather_model.pth` - state dict of the best validation checkpoint.
- `final_weather_model.pth` - full pickled model used by `Test.py`.
- `weather_data/` - 1,123 labelled images. Class is taken from the file name (for example `cloudy12.jpg`): 300 cloudy, 214 rain, 252 shine, 357 sunrise.
- `Weather Recognition.mp4` - demo video.

## Requirements

Imports seen in the code: `torch`, `torchvision`, `cv2` (opencv-python), `numpy`, `PIL`, `matplotlib`, and `re`, `os`, `random`, `gc` from the standard library.

```bash
pip install torch torchvision opencv-python numpy pillow matplotlib
```

## Installation

```bash
git clone <this-repo-url>
cd Weather-Recognition
pip install torch torchvision opencv-python numpy pillow matplotlib
```

A CUDA GPU is used automatically if available; otherwise the code runs on CPU.

## Usage

Run from this folder, because `kaka.py` reads `./weather_data` by relative path.

```bash
# Retrain (overwrites best_weather_model.pth and final_weather_model.pth)
python kaka.py

# Predict on one image
# First edit line 'test_image_path' in Test.py (currently a path on another machine)
python Test.py
```

`Test.py` opens an OpenCV window and waits for a key press before closing.

## Notes

- `kaka.py` uses an unseeded random shuffle, so split contents and test accuracy change between runs.
- The loss plot in `kaka.py` draws the last epoch's average loss as a flat line, not a per-epoch curve.
- `kaka.py` calls `plt.show()`, which blocks until the window is closed.
- `Test.py` loads a full pickled model with `torch.load`. On recent PyTorch versions, loading may require `weights_only=False`. The `CNN` class is defined in `Test.py` so it can be unpickled.
- `Test.py` uses `Resize((150, 150))` to match training. Training resizes with OpenCV before the transform; the two paths are close but not identical.
- The image set is small and four-class; results on other photos are not measured here.
