from ultralytics import YOLO

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Deteksi pose pada video
results = model("WIN_20250311_14_05_30_Pro.mp4", save=True, show=True)