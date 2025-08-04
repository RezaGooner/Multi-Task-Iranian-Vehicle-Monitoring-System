# 🚗 Multi-Task Vehicle Monitoring System

A real-time, Python-based toolkit for intelligent vehicle analytics:
- Vehicle detection  
- Brand & model classification  
- Color recognition  
- Persian (Iranian) license-plate OCR + city lookup  
- Accident detection with audio alerts  

Built on YOLO11 & YOLO8 & ResNet18, OpenCV, and PyTorch. Ideal for smart surveillance, traffic analysis, parking management, and smart-city applications.

---

## 🦾 Key Features

- **Vehicle Detection**  
  • YOLO11 car detector  
- **Brand & Model Classification**  
  • ResNet18 classifier trained on 27 Iranian makes/models  
- **Color Recognition**  
  • ResNet18 classifier for 12 common car colors  
  • Color-coded bounding boxes & labels  
- **License-Plate Recognition (ANPR)**  
  • YOLO11 plate detector + character detector  
  • Parses 8-character Persian/Iranian plates  
  • Looks up city/region from plate data   
- **Accident Detection & Alert**  
  • YOLO11 accident detector  
  • Red bounding box + “Accident (0.XX)” label  
  • Plays configurable beep (`beep.mp3`) at alert intervals  
- **Real-Time Image & Video Support**  
  • Live FPS counter  
  • Saves annotated images & video clips with timestamped filenames  
- **Modular & Configurable**  
  • Centralized `PATHS`, `THRESHOLDS`, `IMG_SIZE`, etc.  
  • Easily swap models or adjust thresholds  
- **Classes of cars**
  ```
    "Arisan", "Atlas", "Dena", "L90", "Mazda vanet", "Megan", "Neissan",
    "Pars", "206", "206 SD", "207", "405", "Peykan", "Pride", "Pride vanet",
    "Pride 111", "Quik", "Rana", "Rio", "Saina", "Samand", "Shahin", "Soren",
    "Tara", "Tiba", "Tiba 2", "Zantia"
  ```
---

## 🚩 Getting Started

### Prerequisites

- Python 3.8+  
- (Optional) CUDA-enabled GPU for real-time performance  

### Install Dependencies

```bash
pip install torch torchvision ultralytics opencv-python pandas playsound pillow numpy
```

> It’s recommended to use a virtual environment.

### Prepare Model Weights

 `weights/` directory has:

```
weights/
├── car_det_model.pt
├── plate_det_model.pt
├── car_name_model.pth
├── color_model.pt
├── char_model.pt
└── accident_model.pt
```

### Model accuracy of machine name determination

```
                precision    recall  f1-score   support

        Arisan      0.884     0.910     0.897       267
         Atlas      0.960     0.930     0.944       128
          Dena      0.990     0.963     0.976      1007
           L90      0.439     0.795     0.566        73
    Mazda-2000      0.997     0.995     0.996       875
         Megan      0.874     0.854     0.864       130
       Neissan      0.995     0.994     0.995       878
  Peugeot Pars      0.997     0.977     0.987      1685
   Peugeot-206      0.987     0.940     0.963      1658
Peugeot-206 SD      0.614     0.780     0.687       173
   Peugeot-207      0.999     0.988     0.994      1632
   Peugeot-405      0.991     0.972     0.981      1299
        Peykan      0.995     0.993     0.994      1063
         Pride      0.991     0.979     0.985      1402
     Pride 111      0.992     0.977     0.984      1216
   Pride vanet      0.764     0.915     0.833       117
          Quik      0.994     0.982     0.988      1429
          Rana      0.949     0.943     0.946       335
           Rio      0.985     0.995     0.990       200
         Saina      0.877     0.916     0.896       179
        Samand      0.996     0.990     0.993      1735
        Shahin      0.655     0.907     0.761        86
         Soren      0.711     0.869     0.782        99
          Tara      0.929     0.929     0.929       183
          Tiba      0.895     0.873     0.884       157
        Tiba 2      0.985     0.981     0.983      1193
        Zantia      0.434     0.930     0.592        57

      accuracy                          0.970     19256
     macro avg      0.884     0.936     0.903     19256
  weighted avg      0.976     0.970     0.972     19256
```

```
Accuracy: 0.9699314499376818
Precision (macro): 0.8844069607888646
Recall (macro): 0.9361670722807696
F1 (macro): 0.9033185541147517
```

<img width="1233" height="931" alt="download" src="https://github.com/user-attachments/assets/bbbb2095-ea7c-4798-a861-aaefcd2c34de" />


### License-Plate City Data

 `Plates/city_plateinfo.txt` (UTF-8 CSV) format:

```
letter,number,city
a,11,Tehran
b,12,Isfahan
...
```

### Sound Alert

Place your beep file at `sound/beep.mp3`.

---

## ⚙ Configuration Overview

All major settings live in `img & video detection.py`:

```python
PATHS = {
  "car_det_model":   "weights/car_det_model.pt",
  "plate_det_model": "weights/plate_det_model.pt",
  "car_name_model":  "weights/car_name_model.pth",
  "color_model":     "weights/color_model.pt",
  "char_model":      "weights/char_model.pt",
  "accident_model":  "weights/accident_model.pt",
  "city_plateinfo":  "Plates/city_plateinfo.txt",
  "alert_sound":     "sound/beep.mp3",
  "source":          "INPUT.JPG",    # or "input.mp4" / camera index
  "output":          "output",
}

IMG_SIZE = (220, 165)     # ResNet image size
THRESHOLDS = {
  "car_name": 0.65,
  "car_color": 0.70,
  "accident": 0.50
}
FONT_SCALE = 0.5
THICKNESS = 1
ALERT_INTERVAL_SEC = 2   # Min seconds between beeps
```

---

## 🚘 Usage

### Process a Single Image

```bash
python img & video detection.py
# Make sure PATHS["source"] points to .jpg/.png
```

- Displays annotated image  
- Saves to `output/output_YYYYMMDDhhmmss.jpg`

### Process Video or Live Stream

```bash
python img & video detection.py
# Set PATHS["source"] = "input.mp4" or camera index (0, 1, …)
```

- Opens a real-time window with FPS  
- Saves annotated video to `output/output_YYYYMMDDhhmmss.mp4`  
- Press **q** to quit

> ### 🎥 Real-Time Webcam Inference
> You can run live vehicle detection, recognition, and analytics using your computer’s webcam. The system will process each video frame in real-time, annotate detections and predictions, and (optionally) record > the result as a video file.

### How to Run on Webcam
  1-Set the Source

  In ```Live.py``` , set the video source to your webcam.
  ```
cap = cv2.VideoCapture(0) 
```
> line 261, 0 is default webcam

2- Run the Script

For the default webcam:
```
python Live.py
```
---

## 📂 Directory Structure

```
.
├── Live.py
├── img & video detection.py
├── Fine-tune/
│   └── car name model.ipynb
├── weights/                   # Model checkpoint files (6 files)
├── Plates/
│   └── city_plateinfo.txt
├── sound/
│   └── beep.mp3
├── INPUT.JPG / input.mp4      # Example inputs
├── output/                    # Auto-created outputs
└── README.md
```

---

## 💡 Output Samples

Below are example outputs from the system.  

### Example video Output

click >> [video output](output/output_20250803220956.mp4)

- Vehicle detection in video
- Vehicle brand and color recognition
- License plate recognition

---

### Example output with overlapping vehicles

| ![o1](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/ae79bc90a4abdc11ba2d867e149bfed64a86e157/output/output_20250804010137.jpg) | ![o2](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/ae79bc90a4abdc11ba2d867e149bfed64a86e157/output/output_20250804010230.jpg) |
|:---:|:---:|
| Vehicle detection in the presence of occlusion | Overlapping vehicles detection |

---

### Example output for license plate reading

| ![o3](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/ae79bc90a4abdc11ba2d867e149bfed64a86e157/output/output_20250804010318.jpg) | ![o4](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/ae79bc90a4abdc11ba2d867e149bfed64a86e157/output/output_20250804010414.jpg) |
|:---:|:---:|
| Accurate license plate recognition | License plate city identification (Iranian plate) |

| ![o5](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/ae79bc90a4abdc11ba2d867e149bfed64a86e157/output/output_20250804010441.jpg) | 
|:---:|
| Another example of license plate reading |  

---

### Example output for detection of traffic accidents

| ![o6](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/b6b301e956f9454704138264a91509db8f2a0d20/output/output_20250804015101.jpg) |
|:---:|
| Accident detection in real-time traffic footage |

---

## 📚 Datasets & Data Collection

Our models were trained on a combination of:

- **Custom Scraped Images** from Iranian automotive platforms:  
  - Iran Khodro (ایران خودرو)  
  - Saipa (سایپا)  
  - Divar (دیوار)  
  - Bama (باما)  
  - Other online classifieds & dealer sites  
- **Public Roboflow Collections**  
  - Car detection (various makes/models)  
  - License-plate detection & OCR  
  - Accident/incident examples  
- **Annotation & Augmentation**  
  - YOLO-formatted bounding boxes & labels  
  - Standardized color, model & plate classes  
  - Augmentations: scaling, rotation, lighting, blur  

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/xyz`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to your fork (`git push origin feature/xyz`)  
5. Open a Pull Request  

---

##  License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🔖 Topics & Keywords

`vehicle-detection` · `car-recognition` · `car-color-detection` · `license-plate-recognition` · `persian-plates` · `anpr` · `accident-detection` · `yolo11` · `pytorch` · `opencv` · `deep-learning` · `smart-city` · `traffic-analysis` · `multitask-learning`

---

> Github.com/RezaGooner

