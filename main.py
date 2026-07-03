import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load model
model = YOLO("best1.pt")

video_path = "test_video.mp4"

cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()

if not ret:
    print("Video open nahi hui")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output_opticalflow.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame)

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls]

            # ROI
            roi_prev = prev_gray[y1:y2, x1:x2]
            roi_curr = gray[y1:y2, x1:x2]

            status = "Static"
            color = (0, 255, 0)

            if roi_prev.size > 0 and roi_curr.size > 0:

                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        roi_prev,
                        roi_curr,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )

                    mag, _ = cv2.cartToPolar(
                        flow[..., 0],
                        flow[..., 1]
                    )

                    avg_motion = np.mean(mag)

                    if avg_motion > 2:
                        status = "Moving"
                        color = (0, 0, 255)

                except:
                    pass

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{class_name} {conf:.2f} | {status}"

            cv2.putText(frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    out.write(frame)
    cv2.imshow("YOLO + Optical Flow", frame)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as:", output_path)