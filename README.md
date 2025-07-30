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

---

## 📂 Directory Structure

```
.
├── main.py
├── weights/               # Model checkpoint files (6 files)
├── Plates/
│   └── city_plateinfo.txt
├── sound/
│   └── beep.mp3
├── INPUT.JPG / input.mp4  # Example inputs
├── output/                # Auto-created outputs
└── README.md
```

---

## 💡 Output Samples

Below are example outputs from the system.  

### Example video Output

click >> [video output](output/output_20250730095049.mp4)

- Vehicle detection in video
- Vehicle brand and color recognition
- License plate recognition

---

### Example output with overlapping vehicles

| ![o1](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/b17f628e50095d26d2560294e20ad9a6017a385d/output/output_20250730132949.jpg) | ![o2](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/b17f628e50095d26d2560294e20ad9a6017a385d/output/output_20250730133041.jpg) |
|:---:|:---:|
| Vehicle detection in the presence of occlusion | Overlapping vehicles detection |

---

### Example output for license plate reading

| ![o3](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/b17f628e50095d26d2560294e20ad9a6017a385d/output/output_20250730133132.jpg) | ![o4](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/b17f628e50095d26d2560294e20ad9a6017a385d/output/output_20250730133216.jpg) |
|:---:|:---:|
| Accurate license plate recognition | License plate city identification (Iranian plate) |

| ![o5](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/b17f628e50095d26d2560294e20ad9a6017a385d/output/output_20250730141531.jpg) | 
|:---:|
| Another example of license plate reading |  

---

### Example output for detection of traffic accidents

| ![o6](https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System/blob/376c4a0cb2dd9693eda4a203870f2eb9de70a6e5/output/output_20250730143116.jpg) |
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

