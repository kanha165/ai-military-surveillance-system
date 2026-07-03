# AI-Powered Military Surveillance System

Real-time military object detection system using YOLOv8 and Optical Flow.

## Features

- Real-time military asset detection
- Detect tanks, missiles, soldiers, helicopters, etc.
- Optical Flow based motion analysis
- Video inference and output video saving
- Bounding box visualization with confidence scores

## Technologies Used

- Python
- OpenCV
- YOLOv8
- Optical Flow
- NumPy

## Project Structure

```text
military_detection/
│
├── main.py
├── best1.pt
├── test_video.mp4
├── output_video.mp4
├── BoxF1_curve.png
├── BoxPR_curve.png
├── confusion_matrix.png
└── results.png
```

## Installation

```bash
pip install ultralytics opencv-python numpy
```

## Run

```bash
python main.py
```

## Sample Results

Model detects:

- Tank
- Missile
- Soldier
- Military Vehicle
- Helicopter

## Future Improvements

- Object Tracking
- Speed Estimation
- Heatmap Generation
- Intrusion Detection
- Alert System

## Author

Kanha Patidar