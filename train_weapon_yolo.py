from ultralytics import YOLO
import os

DATA_YAML = "dataset1/data.yaml"
BASE_MODEL = "yolov8n.pt"   # FIXED (use YOLOv8)
EPOCHS = 20                # increased for better accuracy
IMG_SIZE = 640             # better detection
BATCH_SIZE = 8             # increase if RAM allows

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

print("Dataset found.")
print("Starting YOLO Weapon Detection Training...\n")

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name="weapon_detector",
    workers=2,
    device="cpu",
    patience=10
)

print("\nTraining Finished!")

metrics = model.val()
print("\nValidation Results:")
print(metrics)

print("\nBest model saved at:")
print("runs/detect/weapon_detector/weights/best.pt")