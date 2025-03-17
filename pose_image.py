from ultralytics import YOLO
# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Deteksi pose pada gambar
results = model("https://ultralytics.com/images/bus.jpg", show=True)

# Simpan hasil
results[0].save("pose_result.jpg")