#!/usr/bin/env python3
import click
from pathlib import Path
from src import (
    My_LicensePlate_Model,
    log,
    process_video,
)
from src.train import train as train_func


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """License Plate Detection System using YOLO."""
    pass


@cli.command("train-cmd")
@click.option("--data", "-d", default="data/data.yaml", type=click.Path(), help="Path to data.yaml")
@click.option("--model", "-m", default="yolov8n.pt", help="Pretrained model name or path")
@click.option("--epochs", "-e", default=30, type=int, help="Number of epochs")
@click.option("--imgsz", "-s", default=640, type=int, help="Image size")
@click.option("--batch", "-b", default=8, type=int, help="Batch size")
@click.option("--device", default=0, help="Device (0 for GPU, 'cpu' for CPU)")
@click.option("--project", help="Project directory")
@click.option("--name", "-n", default="train", help="Experiment name")
def train_cmd(data, model, epochs, imgsz, batch, device, project, name):
    train_func(
        data_yaml=data,
        model_name=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )


@cli.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input video file path")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output video file path")
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to model weights (.pt)")
@click.option("--conf", "-c", default=0.35, type=float, help="Confidence threshold for drawing")
@click.option("--infer-conf", default=0.15, type=float, help="Confidence for model inference")
@click.option("--max-det", default=50, type=int, help="Max detections per frame")
@click.option("--imgsz", default=None, type=int, help="Inference image size")
@click.option("--augment/--no-augment", default=False, help="Use test-time augmentation")
@click.option("--smooth/--no-smooth", default=True, help="Use temporal smoothing")
@click.option("--show/--no-show", default=False, help="Show preview window")
def video(input, output, model, conf, infer_conf, max_det, imgsz, augment, smooth, show):
    """Process video file for license plate detection."""
    log.info(f"CLI: video mode - input={input}, output={output}")
    process_video(
        input_path=input,
        output_path=output,
        model_path=model,
        conf_threshold=conf,
        infer_conf=infer_conf,
        max_det=max_det,
        imgsz=imgsz,
        augment=augment,
        use_smoothing=smooth,
        show_preview=show,
    )


@cli.command()
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to model weights (.pt)")
def info(model):
    detector = My_LicensePlate_Model(model) if model else My_LicensePlate_Model()
    info = detector.get_model_info()
    click.echo(f"Model path: {info['model_path']}")
    click.echo(f"Model name: {info['model_name']}")
    click.echo(f"Classes: {info['classes']}")


if __name__ == "__main__":
    cli()