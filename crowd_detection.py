from ultralytics import YOLO
import cv2
import time

# ==========================================
# Load YOLOv8 Model
# ==========================================
model = YOLO("yolov8s.pt")

# ==========================================
# Open Video
# ==========================================
video_path = "videos/factory.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video Finished.")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # ==========================================
    # YOLO Detection
    # ==========================================
    results = model(frame, verbose=False)[0]

    worker_count = 0
    total_worker_area = 0

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Detect only PERSON
        if cls != 0:
            continue

        if conf < 0.50:
            continue

        worker_count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1

        total_worker_area += width * height

        # -----------------------------
        # Worker Bounding Box
        # -----------------------------
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Small Filled Label
        cv2.rectangle(
            frame,
            (x1, y1 - 22),
            (x1 + 65, y1),
            (0, 150, 0),
            -1
        )

        cv2.putText(
            frame,
            "Worker",
            (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # ==========================================
    # Crowd Occupancy
    # ==========================================
    occupancy = (total_worker_area / frame_area) * 100

    # ==========================================
    # Crowd Density Logic
    # ==========================================
    if occupancy < 18:
        density = "LOW"
        risk = "SAFE"
        color = (0, 255, 0)

    elif occupancy < 35:
        density = "MEDIUM"
        risk = "WARNING"
        color = (0, 255, 255)

    else:
        density = "HIGH"
        risk = "CRITICAL"
        color = (0, 0, 255)

    # ==========================================
    # FPS
    # ==========================================
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # ==========================================
    # Transparent Dashboard
    # ==========================================
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (430, 190),
        (35, 35, 35),
        -1
    )

    alpha = 0.45

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )

    # Dashboard Border
    cv2.rectangle(
        frame,
        (10, 10),
        (430, 190),
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Industrial Crowd Monitoring",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Workers Detected : {worker_count}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Crowd Occupancy : {occupancy:.1f} %",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Crowd Density : {density}",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Risk Level : {risk}",
        (25, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    cv2.putText(
        frame,
        f"FPS : {fps:.1f}",
        (320, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # ==========================================
    # Transparent Alert Banner
    # ==========================================
    if risk == "CRITICAL":

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (450, 15),
            (950, 70),
            (0, 0, 255),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.45,
            frame,
            0.55,
            0,
            frame
        )

        cv2.rectangle(
            frame,
            (450, 15),
            (950, 70),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "OVERCROWDING DETECTED",
            (470, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    # ==========================================
    # Display
    # ==========================================
    cv2.imshow("Industrial Crowd Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()