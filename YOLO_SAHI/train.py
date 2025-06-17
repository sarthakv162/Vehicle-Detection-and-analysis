from ultralytics import YOLO
import torch
import yaml


def train_yolov8(
    data_yaml_path,
    model_size='n',  # n, s, m, l, x
    epochs=50,
    batch_size=8,
    img_size=640,
    pretrained_weights=None
):
    """
    Fine-tune YOLOv8 model on custom dataset
    """
    # Check GPU availability
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == '0':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load a model
    if pretrained_weights:
        model = YOLO(pretrained_weights)
    else:
        model = YOLO(f'yolov8{model_size}.pt')  # load a pretrained model

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=5,  # early stopping patience
        save=True,  # save best checkpoint
        device=device,  # use GPU if available
        workers=4,  # number of worker threads
        amp=True  # enable automatic mixed precision
    )
    
    return results



if __name__ == "__main__":
    data_yaml_path = "dataset/data.yaml"


    train_yolov8(data_yaml_path=data_yaml_path)
