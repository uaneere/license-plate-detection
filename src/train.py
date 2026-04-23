from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({"datasets_dir": "."})


def train(
    data_yaml: str = "data/data.yaml",
    model_name: str = "yolo11l.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    device: int = 0,
    project: str = "runs/detect",
    name: str = "train",
):

    model = YOLO(model_name)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )


if __name__ == "__main__":
    train()