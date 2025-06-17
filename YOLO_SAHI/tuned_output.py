from ultralytics import YOLO
import os
import torch
import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import cv2
import numpy as np


def train_yolov8(
    data_yaml_path,
    model_size='n',  # n, s, m, l, x
    epochs=50,
    batch_size=8,
    img_size=640,
    pretrained_weights=None  # Changed from True to None to use default pretrained weights
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

def run_inference(weights_path, data_yaml_path, conf_threshold=0.25, save_dir='tuned_inference_results'):
    """
    Run inference on validation set using trained weights with SAHI for small vehicle detection
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val_results'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val_results', 'labels'), exist_ok=True)
    
    # Load the trained model
    model = YOLO("best.pt")
    
    # Initialize SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path="best.pt",
        confidence_threshold=conf_threshold,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # Load data configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get validation images path directly from yaml
    val_path = data_config['val']
    
    print(f"Running inference on validation set at: {val_path}")
    print(f"Class names: {data_config['names']}")
    
    # Process each image in the validation set
    for img_name in os.listdir(val_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(val_path, img_name)
            
            # Read image
            image = read_image(img_path)
            
            # Get SAHI predictions
            result = get_sliced_prediction(
                image=image,
                detection_model=detection_model,
                slice_height=96,
                slice_width=192,
                overlap_height_ratio=0.6,
                overlap_width_ratio=0.6,
            )
            
            # Convert predictions to YOLO format
            predictions = result.object_prediction_list
            
            # Draw predictions on image
            img = cv2.imread(img_path)
            for pred in predictions:
                bbox = pred.bbox.to_xyxy()
                conf = pred.score.value
                class_id = pred.category.id
                
                # Draw bounding box
                cv2.rectangle(img, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 2)
                
                # Add label
                label = f"{data_config['names'][class_id]} {conf:.2f}"
                cv2.putText(img, label, 
                          (int(bbox[0]), int(bbox[1]-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 0), 2)
                
                # Save detection in YOLO format
                # Convert bbox to YOLO format (x_center, y_center, width, height)
                img_height, img_width = img.shape[:2]
                x_center = (bbox[0] + bbox[2]) / 2 / img_width
                y_center = (bbox[1] + bbox[3]) / 2 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                
                # Save to txt file
                txt_path = os.path.join(save_dir, 'val_results', 'labels', 
                                      os.path.splitext(img_name)[0] + '.txt')
                with open(txt_path, 'a') as f:
                    f.write(f"{class_id} {x_center} {y_center} {width} {height} {conf}\n")
            
            # Save annotated image
            cv2.imwrite(os.path.join(save_dir, 'val_results', img_name), img)
    
    print(f"Inference completed. Results saved in {os.path.join(save_dir, 'val_results')}")
    return True

if __name__ == "__main__":
    data_yaml_path = "dataset/data.yaml"
    
    # Run inference with SAHI
    results = run_inference(
        weights_path="best.pt",
        data_yaml_path=data_yaml_path,
        conf_threshold=0.1,  # confidence threshold
        save_dir='tune_inference_results_1'
    )

# for tune_inference_result 1 , we did 256 px with 0.2 overlap and 0.25 conf threshold and added NMS 
# for tune_inference_result 2 , we did 128-256 px with 0.25 overlap , 0.1 conf and without NMS
# for tune_inference_result 3 , we did 160-320 px with 0.3 overlap , 0.2 conf threshold   
# Usually NMS is not needed here because yolo internally applies and returns less non overlapping detections 
