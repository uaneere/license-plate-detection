# License Plate Detection System

**Student:** Ruslan Agafonov & Angelina Chernikova

**Group:** 972401

## Description

License plate detection system using YOLO

## Project Structure

```
license-plate-detection/
├── src/
│   ├── __init__.py
│   ├── logger.py          # Logging (singleton)
│   ├── model_impl.py      # My_LicensePlate_Model class
│   ├── video_mode.py      # Video file processing
│   ├── track_smoothing.py # Temporal smoothing
│   ├── visualization.py   # Bounding box rendering
│   └── train.py           # Model training
├── scripts/
│   └── split_dataset.py   # Dataset splitting
├── main.py                # CLI entry point
├── pyproject.toml         # Poetry config
├── dist/
│   └── *.whl              # Built package
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation with Poetry

### Prerequisites

```bash
pip install poetry
```

### Install dependencies

```bash
git clone https://github.com/uaneere/license-plate-detection.git
cd license-plate-detection

poetry install

poetry run pip install ultralytics
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
poetry run python -c "import torch; print(torch.cuda.is_available())"

poetry env activate
```

## Building .whl Package

```bash
poetry build
```

### What's inside .whl?

```
src/__init__.py
src/logger.py
src/model_impl.py
src/video_mode.py
src/train.py
main.py
```

- ❌ No model weights (.pt)
- ❌ No dataset (data/)
- ❌ No venv
- ❌ No runs/

## Dataset Preparation

### Option 1: Use ready dataset

```bash
# Download from Roboflow
# Place images and labels in data/train/ and data/valid/
# Create data/data.yaml
```

### Option 2: Create your own dataset

```bash
# Markup video with Roboflow, Label Studio, or CVAT
# Export in YOLO format
# Split dataset
python scripts/split_dataset.py
```

### data.yaml structure

```yaml
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['License-Plate']
```

## Training

```bash
poetry run python main.py train-cmd --data data/data.yaml --epochs 30 --batch 8
```

### Training results

| Metric     | Value  |
|------------|--------|
| mAP50      | 0.743  |
| mAP50-95   | 0.412  |
| Precision  | 0.853  |
| Recall     | 0.701  |

## Usage

### Video processing

```bash
poetry run python main.py video \
  -i test.mp4 \
  -o output.mp4 \
  -m runs/detect/train/weights/best.pt \
  --conf 0.05 --smooth --show 
  # модель получилась слишком неуверенная, работает хорошо при низком conf <0.1, закибербуллили
```

### Model info

```bash
poetry run python main.py info -m runs/detect/train/weights/best.pt
```

## Docker

### Add your test data in directory "input"

### Run video processing

```bash
# Use one of these methods
docker build -t lpd .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/input:/app/input \
  lpd \
  video -i /app/input/test.mp4 -o /app/output/result.mp4 -m /app/runs/detect/train/weights/best.pt --conf 0.05

docker compose run --rm app video -i /app/input/test.mp4 -o /app/output/result.mp4 -m /app/runs/detect/train/weights/best.pt --conf 0.05
```


## Install from .whl

```bash
pip install dist/license_plate_detection-0.1.0-py3-none-any.whl

lpd video -i input.mp4 -o output.mp4 -m best.pt
```

## Data resources

https://youtu.be/dQw4w9WgXcQ?si=5XVbnMgDK9blrGiR
