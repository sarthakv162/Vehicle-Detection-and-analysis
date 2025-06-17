# Traffic Analysis System using YOLOv8 and SAHI

This project implements a comprehensive traffic analysis system using YOLOv8 for object detection and SAHI (Slicing Aided Hyper Inference) for improved detection performance. The system provides real-time traffic analysis including vehicle counting, speed estimation, and traffic density calculation.

## Project Structure

```
├── yolo-V8/                    # Original YOLOv8 model implementation
├── yolo_inference/             # Inference results from finetuned YOLOv8-nano model
├── analysis.py                 # Main traffic analysis script
├── visuals.py                  # Visualization and metrics calculation utilities
├── tuned_output.py             # SAHI implementation for inference
├── train.py                    # YOLOv8 model finetuning script
├── best.pt                     # Finetuned YOLOv8 model weights
├── yolov8n.pt                  # Original YOLOv8-nano weights
```

## Key Features

- Real-time vehicle detection using YOLOv8-nano
- Vehicle tracking using DeepSORT
- Traffic analysis features:
  - Live vehicle counting (up/down direction)
  - Vehicle speed estimation
  - Traffic density calculation
- SAHI Vision Library implementation for improved detection performance
- Model finetuning capabilities

## Components

### Model Training and Inference
- `train.py`: Script for finetuning YOLOv8 model on custom dataset
- `tuned_output.py`: Implements SAHI for improved inference results
- `best.pt`: Finetuned model weights
- `yolov8n.pt`: Original YOLOv8-nano weights

### Analysis and Visualization
- `analysis.py`: Main traffic analysis script implementing:
  - Vehicle counting
  - Speed estimation
  - Traffic density calculation
  - DeepSORT tracking
- `visuals.py`: Utilities for visualization and metrics calculation

## Usage

1. Model Training:
```bash
python train.py
```

2. Traffic Analysis:
```bash
python analysis.py
```

3. SAHI Inference:
```bash
python tuned_output.py
```

## Dependencies

- YOLOv8
- SAHI
- DeepSORT
- OpenCV
- NumPy
- PyTorch