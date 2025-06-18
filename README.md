
# Vehicle Detection and analysis

This repository contains implementations of various object detection and vehicle detection models evaluated on a custom traffic dataset. Models range from lightweight two-stage detectors to modern CNN and transformer-based architectures, as well as a self-supervised pretraining pipeline.

---

## 📁 Directory Structure

```

├── Faster-rcnn-mobilenet/           # Faster R-CNN with MobileNet backbone
├── FasterRCNN RESNET 50/            # Faster R-CNN with ResNet-50 backbone
├── MASK RCNN MOBNET/                # Mask R-CNN with MobileNet backbone (blank pretrain)
├── MASK RCNN MOBNET mask/           # Mask R-CNN with MobileNet backbone (masked pretrain)
├── MASK RCNN RESNET 50/             # Mask R-CNN with ResNet-50 backbone (blank pretrain)
├── MASK RCNN RESNET 50 mask/        # Mask R-CNN with ResNet-50 backbone (masked pretrain)
├── MASK RCNN RESNET 101/            # Mask R-CNN with ResNet-101 backbone (blank pretrain)
├── MASK RCNN RESNET 101 mask/       # Mask R-CNN with ResNet-101 backbone (masked pretrain)
├── MASK RCNN ResNeXt 101 mask/      # Mask R-CNN with ResNeXt-101 backbone (masked pretrain)
├── SIMSIAM RetinaNet/               # SimSiam self-supervised pretrain + RetinaNet fine-tune
├── YOLO\_SAHI/                      # YOLOv5/YOLOv8 with SAHI slicing for improved small-object detection
├── Yolo + diff detections/          # Yolo Model with Different Detection methords
└── detr-try/                        # DETR and Sparse R-CNN experiments


````

---

## 🔧 Requirements

- Python 3.8+  
- PyTorch 1.10+  
- torchvision  
- Detectron2 (for some Mask R-CNN implementations)  
- [YOLOv5/YOLOv8 dependencies](https://github.com/ultralytics/yolov5)  
- SAHI (Slicing Aided Hyper Inference)  
- COCO API  
- albumentations (data augmentation)  
- timm (for backbone architectures)  

Install via:
```bash
pip install torch torchvision detectron2 pycocotools albumentations timm sahi
# Plus YOLO requirements
pip install -r YOLO_SAHI/requirements.txt
````

---

## 📂 Data Preparation

1. Place your dataset in COCO format under `data/traffic_dataset/`:

   * `images/`: All JPEG/PNG images.
   * `annotations/train.json`, `annotations/val.json`.
2. Update paths in each model's config or script as needed.

---

## ▶️ Running Training and Inference

Each folder contains:

* `train.py`: Script to train the model on your dataset.
* `inference.py`: Script to run inference on validation/test images.
* `config.yaml` or `config.py`: Model and dataset configurations.

Example (ResNet-50 Faster R-CNN):

```bash
cd "FasterRCNN RESNET 50"
python train.py --config config.yaml
python inference.py --weights outputs/model_final.pth --images ../data/traffic_dataset/images/val/
```

---

## Results Summary

| Model                                     | mAP @ .50:.95 | mAP @ .50 |
| ----------------------------------------- | ------------- | --------- |
| Mask R‑CNN (MobileNet, blank pretrain)    | 0.510         | 0.858     |
| Mask R‑CNN (MobileNet, masked pretrain)   | 0.532         | 0.866     |
| Mask R‑CNN (ResNet‑50, blank pretrain)    | 0.669         | 0.941     |
| Mask R‑CNN (ResNet‑50, masked pretrain)   | 0.661         | 0.921     |
| Mask R‑CNN (ResNet‑101, blank pretrain)   | 0.673         | 0.932     |
| Mask R‑CNN (ResNet‑101, masked pretrain)  | 0.643         | 0.929     |
| Mask R‑CNN (ResNeXt‑101, masked pretrain) | 0.643         | 0.929     |
| SSVL (SimSiam) + RetinaNet                | 0.580         | 0.915     |
| Faster R‑CNN (ResNet‑50)                  | 0.678         | 0.946     |
| YOLO + SAHI                               |  0.663        | 0.889     |
| Faster R‑CNN (MobileNet)                  | 0.310         | 0.594     |
| DETR (200 epochs)                         | 0.007         | 0.009     |
| Sparse R‑CNN                              | 0.005         | 0.021     |

Refer to the [results notebook](./results/analysis.ipynb) for detailed metrics and plots.

---

## Customization

* **Backbones**: Swap out backbone networks via the config files (e.g., ResNet, ResNeXt, MobileNet).
* **Pretraining**: Use masked vs. blank pretraining by toggling flags in `train.py`.
* **Data Augmentation**: Customize `albumentations` pipelines in `files`.
* **Inference Slicing**: Adjust SAHI parameters for small object detection in `YOLO_SAHI`.

---

## Contributing

Feel free to open issues or pull requests to improve models, add new architectures, or enhance data pipelines.

---

## Contact

— **Sarthak Verma**  
- LinkedIn: [linkedin](https://www.linkedin.com/in/sarthak-verma-6002001b4/)  
- GitHub: [github](https://github.com/sarthakv162/)  




