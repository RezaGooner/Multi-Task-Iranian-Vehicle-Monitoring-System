---
Title: 'Multi-task Iranian Vehicle Surveillance System: An Open Source Deep Learning Framework for Vehicle License Plate Recognition, Classification and Recognition'
Tags:
 - Python
 - Deep Learning
 - Computer Vision
 - Multi-task Learning
 - License Plate Recognition
 - Vehicle Recognition
Authors:
 - Name: Reza Asadi
orcid: 0009-0005-6852-5756
Affiliation: 1
Affiliations:
 - Name: "Faculty of Computer Engineering, Yasuj University, Yasuj, Iran"
Citation: 1
Date: 2025-09-09
Bibliography: paper.bib
---

# Summary

**Multi-task Iranian Vehicle Surveillance System** is an open source deep learning framework designed for simultaneous vehicle recognition, vehicle type classification and license plate recognition in Iranian traffic environments. Leveraging advanced convolutional neural networks and multi-task learning techniques, the system addresses the challenges of real-time Intelligent Transportation Systems (ITS) by providing an integrated, modular, and scalable pipeline.

The software helps researchers, traffic enforcement organizations, and smart city developers by providing a repeatable platform for vehicle monitoring with pre-trained models on region-specific datasets, focusing on the unique design, typography, and color patterns of Iranian license plates.

# Statement of need

Vehicle monitoring in Iran presents unique technical challenges due to the diversity of license plate formats, ambient lighting, camera angles, and vehicle types. Existing solutions often specialize in only one of the following: license plate recognition, classification, or recognition, and are optimized for non-Iranian conditions. This project fills this gap by introducing a multi-task architecture that jointly trains and executes these tasks in real time.

This project provides the following:
- End-to-end deep learning pipeline optimized for Iranian traffic conditions.

- Modular structure that allows for the replacement of detection, classification or OCR submodules.

- Pre-trained models ready for immediate use and deployment.

# Software Description

## Functionality

The system consists of the following components:

1. **Vehicle Detection Module** - Uses models such as YOLOv5/YOLOv8 to locate vehicles in an image or video frame.

2. **Vehicle Classification Module** - Classifies the detected vehicles into predefined categories (name and model and manufacturer).

3. **License Plate Recognition (LPR) Module** - Detects and recognizes Iranian license plates using a specialized combination of CNN + OCR.

4. **Accident Detection** - Accident detection and voice notification

5. **Car Color Detection**

6. **Detect the place of issuance of car license plates by city**

## Technologies

- **Language:** Python 3.x
- **Libraries:** PyTorch/TensorFlow, OpenCV, NumPy, Pandas, Albumentations
- **Frameworks:** YOLOv5/YOLOv8, CRNN for OCR

# Installation and Usage

To install:
```bash
git clone https://github.com/RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System.git
cd Multi-Task-Iranian-Vehicle-Monitoring-System
```

To run:
```bash
python img & video detection.py --source test_video.mp4
```

# Use cases

- Traffic monitoring and data collection in urban areas. - Automatic toll collection by vehicle and license plate registration.
- Research on multi-task learning in computer vision.

#Acknowledgements

We acknowledge the use of:
- Iranian open dataset for license plate recognition.
- Open source computer vision libraries and YOLO pre-trained models.
- Open source community contributions.

# References
---
