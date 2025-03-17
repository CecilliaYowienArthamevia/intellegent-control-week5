from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import threading

# Load model YOLOv8 untuk deteksi pose dan objek person
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kosongkan buffer kamera untuk menghindari latensi

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Warna dan ketebalan garis
COLOR = (0, 255, 0)
THICKNESS = 2

# Pasangan titik (skeleton) berdasarkan format COCO
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Lengan kanan
    (0, 5), (5, 6), (6, 7),               # Lengan kiri
    (0, 8), (8, 9), (9, 10),              # Kaki kanan
    (0, 11), (11, 12), (12, 13),          # Kaki kiri
    (1, 14), (14, 16)                     # Mata kanan ke telinga kanan
]

# Nama-nama titik berdasarkan format COCO
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# === Fungsi untuk deteksi objek (person) ===
def detect_objects(frame):
    results = object_model(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            if int(class_id) == 0:  # Person
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# === Fungsi untuk deteksi pose (skeleton) ===
def detect_pose(frame):
    results = pose_model(frame)
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        if keypoints is not None:
            for person_keypoints in keypoints:
                for i, (x, y) in enumerate(person_keypoints):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f'{KEYPOINT_NAMES[i]}', (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(frame, f'{i}', (int(x) - 10, int(y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Gambar koneksi antar titik (skeleton)
                for pair in SKELETON_CONNECTIONS:
                    part_a, part_b = pair
                    if part_a < len(person_keypoints) and part_b < len(person_keypoints):
                        x1, y1 = person_keypoints[part_a]
                        x2, y2 = person_keypoints[part_b]
                        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR, THICKNESS)

# === Fungsi untuk deteksi jari (hands) ===
def detect_hands(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(frame, f'{i}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

# === Loop utama ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Jalankan deteksi secara paralel dengan threading
    t1 = threading.Thread(target=detect_objects, args=(frame,))
    t2 = threading.Thread(target=detect_pose, args=(frame,))
    t3 = threading.Thread(target=detect_hands, args=(frame,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    # Tampilkan hasil
    cv2.imshow("YOLOv8 + MediaPipe (Multi-Threading)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
