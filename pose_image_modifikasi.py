from ultralytics import YOLO
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Load model YOLOv8 Pose dan Object
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

# Warna dan ketebalan garis
COLOR = (0, 255, 0)
THICKNESS = 2

# Pasangan titik (skeleton) berdasarkan format COCO
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
    (1, 14), (14, 16)
]

# Nama-nama titik berdasarkan format COCO
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# === Baca gambar dari URL ===
url = "https://ultralytics.com/images/bus.jpg"  # URL gambar
try:
    response = requests.get(url)
    response.raise_for_status()  # Periksa status response
    img_data = BytesIO(response.content)
    img = Image.open(img_data).convert('RGB')
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Konversi ke format BGR untuk OpenCV
except requests.exceptions.RequestException as e:
    print(f"Failed to download image from URL: {e}")
    exit()

# === Deteksi objek (person) dengan YOLOv8 ===
object_results = object_model(frame)

for result in object_results:
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, conf in zip(boxes, class_ids, confidences):
        if int(class_id) == 0:  # class_id = 0 untuk person
            x1, y1, x2, y2 = map(int, box)
            # Buat kotak biru di sekitar person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Tambahkan label "Person"
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# === Deteksi pose tubuh dengan YOLOv8 Pose ===
pose_results = pose_model(frame)

for result in pose_results:
    keypoints = result.keypoints.xy.cpu().numpy()

    if keypoints is not None:
        for person_keypoints in keypoints:
            # Gambar semua titik tubuh untuk setiap orang
            for i, (x, y) in enumerate(person_keypoints):
                if x > 0 and y > 0:
                    # Gambar titik tubuh
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Titik merah
                    # Tambahkan keterangan nama titik
                    cv2.putText(frame, f'{KEYPOINT_NAMES[i]}', (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    # Tambahkan keterangan angka di samping titik
                    cv2.putText(frame, f'{i}', (int(x) - 10, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Gambar semua koneksi antar titik (skeleton)
            for pair in SKELETON_CONNECTIONS:
                part_a, part_b = pair
                if part_a < len(person_keypoints) and part_b < len(person_keypoints):
                    x1, y1 = person_keypoints[part_a]
                    x2, y2 = person_keypoints[part_b]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR, THICKNESS)

# === Simpan hasil ===
output_image = "output_pose_result.jpg"
cv2.imwrite(output_image, frame)
print(f"Hasil disimpan di: {output_image}")

# === Tampilkan hasil ===
cv2.imshow("YOLOv8 Pose and Object Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
