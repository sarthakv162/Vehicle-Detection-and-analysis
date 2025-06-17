import matplotlib.pyplot as plt
import cv2 
import os 
from ultralytics import YOLO
import yaml
import shutil


if __name__ == "__main__":

    # Load YAML config
    with open('dataset/data.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load the model
    model = YOLO('best.pt')

    # Create output directory if it doesn't exist
    output_dir = 'yolo_inference'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run validation and get metrics
    metrics = model.val(data='dataset/data.yaml', split='val')
    print("\nValidation Metrics:")
    print(f"mAP@0.5: {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")

    # Run inference on validation set
    val_path = config['val']
    results = model.predict(val_path, save=True, project=output_dir, name='predictions')

    print(f"\nResults saved in {os.path.join(output_dir, 'predictions')}")

    # Example: visualize 5 frames with boxes
    for img_file in sorted(os.listdir(val_path))[:10]:
        frame = cv2.imread(os.path.join(val_path, img_file))
        results = model.predict(frame, imgsz=640)[0]

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)  # green box

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title(f"Detections in {img_file}")
        plt.axis('off')
        plt.show()
