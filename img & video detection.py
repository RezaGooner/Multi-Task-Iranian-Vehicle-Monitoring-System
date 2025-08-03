# ==============================
#   IMPORT NECESSARY LIBRARIES
# ==============================

import torch
import pandas as pd
from ultralytics import YOLO
from torchvision import models, transforms
import cv2
import threading
from playsound import playsound
import numpy as np
import time
import os
import datetime
from PIL import Image

# ================== CONFIG & PATHS =====================

PATHS = {
    "car_det_model":   "weights/car_det_model.pt",
    "plate_det_model": "weights/plate_det_model.pt",
    "car_name_model":  "weights/car_name_model.pth",
    "color_model":     "weights/color_model.pt",
    "char_model":      "weights/char_model.pt",
    "accident_model":  "weights/accident_model.pt",
    "city_plateinfo":  "Plates/city_plateinfo.txt",
    "alert_sound":     "sound/beep.mp3",
    "source":          "e.jpg",   
    "output":          "output",
}

# ================== SIZE & ARRANGEMENT ==================

IMG_SIZE           = (220, 165)
FONT_SCALE_CAR     = 0.68
FONT_SCALE_COLOR   = 0.68
FONT_SCALE_PLATE   = 0.75
FONT_SCALE_ACC     = 1.1
FONT_THICKNESS     = 1
ALERT_INTERVAL_SEC = 2

# ================== THRESHOLDS ==================

THRESHOLDS = {
    "car_name": 0.65,
    "car_color": 0.7,
    "accident": 0.5,
}

# ================== CAR & COLOR CLASSES =====================

car_classes = [
    "Arisan", "Atlas", "Dena", "L90", "Mazda vanet", "Megan", "Neissan",
    "Pars", "206", "206 SD", "207", "405", "Peykan", "Pride", "Pride vanet",
    "Pride 111", "Quik", "Rana", "Rio", "Saina", "Samand", "Shahin", "Soren",
    "Tara", "Tiba", "Tiba 2", "Zantia"
]
num_car_classes = len(car_classes)

color_class_names = [
    "Black", "Blue", "Brown", "Crismon", "Gray", "Green",
    "Orange", "Purple", "Red", "Silver", "White", "Yellow"
]
num_color_classes = len(color_class_names)

color_bgr = {
    "Black":     (0, 0, 0),
    "Blue":      (255, 0, 0),
    "Brown":     (19, 69, 139),
    "Crismon":   (60, 20, 220),
    "Gray":      (128, 128, 128),
    "Green":     (0, 128, 0),
    "Orange":    (0, 140, 255),
    "Purple":    (128, 0, 128),
    "Red":       (0, 0, 255),
    "Silver":    (192, 192, 192),
    "White":     (255, 255, 255),
    "Yellow":    (0, 255, 255)
}

charclassnames = [
    '0','9','b','d','a','ein','g','gh','h','n','s','1','malul','n','s','sad','t','ta','v','y',
    '2','3','4','5','6','7','8'
]

# ================== DEVICE =====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("github >> REZAGOONER")

# ================== CITY DATA =====================

city_plateinfo = pd.read_csv(PATHS["city_plateinfo"], delimiter=',', encoding='utf-8')

def find_city_by_plate(letter: str, number: str, df: pd.DataFrame) -> str:
    result = df[(df['letter'] == letter) & (df['number'] == number)]
    return result.iloc[0]['city'] if len(result) > 0 else ''

# ================== MODEL LOAD =====================

car_model = models.resnet18(weights=None)
car_model.fc = torch.nn.Linear(car_model.fc.in_features, num_car_classes)
car_model.load_state_dict(torch.load(PATHS["car_name_model"], map_location=device))
car_model = car_model.to(device).eval()
car_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

color_model = models.resnet18(weights=None)
color_model.fc = torch.nn.Linear(color_model.fc.in_features, num_color_classes)
color_model.load_state_dict(torch.load(PATHS["color_model"], map_location=device))
color_model = color_model.to(device).eval()
color_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

car_model_weight      = YOLO(PATHS["car_det_model"])
plate_model_weight    = YOLO(PATHS["plate_det_model"])
char_model_weight     = YOLO(PATHS["char_model"])
accident_model_weight = YOLO(PATHS["accident_model"])

# ================== UTILITIES =====================

def predict_car(car_img):
    try:
        car_pil = Image.fromarray(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB))        
        car_tensor = car_transform(car_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = car_model(car_tensor)
            pred_softmax = torch.softmax(pred, dim=1)
            
            pred_class = pred.argmax(1).item()
            prob = pred_softmax[0, pred_class].item()
            

            return car_classes[pred_class], prob

    except Exception as e:
        print("ERROR in predict_car:", str(e))
        return "UNKNOWN", 0.0

def predict_color(car_img):
    color_pil = Image.fromarray(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB))
    color_tensor = color_transform(color_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = color_model(color_tensor)
        pred_class = pred.argmax(1).item()
        prob = torch.softmax(pred, dim=1)[0, pred_class].item()
        return color_class_names[pred_class], prob

def putText_with_outline(img, text, org, font, font_scale, color, outline_thickness=4, text_thickness=1):
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, text_thickness, cv2.LINE_AA)

def play_alert_sound(path=PATHS["alert_sound"]):
    thr = threading.Thread(target=playsound, args=(path,), daemon=True)
    thr.start()

# =============== MAIN FRAME PROCESSING & LABEL THRESHOLD  ===============

last_alert_time = [0]

vehicle_class_ids = [2, 3, 5, 7] # (car=2), (truck=7), (bus=5), (motorcycle=3)

def process_frame(img):
    global last_alert_time
    result_img = img.copy()
    accident_detected = False

    # ---- CAR DETECTION + CAR NAME & COLOR ----
    out_car = car_model_weight(img, show=False, conf=0.6)
    for det in out_car:
        for box in det.boxes:
            class_id = int(box.cls[0])

            if class_id not in vehicle_class_ids:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            vehicle_img = img[y1:y2, x1:x2]
            box_color = (0, 200, 255)
            label_parts = []
            text_color = (128, 128, 128)

            if vehicle_img.shape[0] > 0 and vehicle_img.shape[1] > 0:
                # only (car=2) 
                if class_id == 2:
                    car_label, car_prob = predict_car(vehicle_img)
                    if car_prob >= THRESHOLDS["car_name"]:
                        label_parts.append(f"{car_label} ({car_prob:.2f})")
                        text_color = (10,120,255)
                # color detrction for all classes
                color_label, color_prob = predict_color(vehicle_img)
                if color_prob >= THRESHOLDS["car_color"]:
                    label_parts.append(f"{color_label} ({color_prob:.2f})")
                    text_color = color_bgr.get(color_label, (128,128,128))
            
            vehicle_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
            label_parts = [vehicle_names[class_id]] + label_parts
            
            label_full = " | ".join(label_parts)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 2)
            if label_full.strip() != "":
                putText_with_outline(
                    result_img, label_full, (x1, y1-9),
                    cv2.FONT_HERSHEY_TRIPLEX, FONT_SCALE_CAR, text_color, 3, FONT_THICKNESS
                )

    # ---- PLATE DETECTION & OCR ----
    out_plate = plate_model_weight(img, show=False, conf=0.6)
    for det in out_plate:
        for box in det.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (130, 0, 255), 2)
            plate_img = img[y1:y2, x1:x2]
            if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                plate_output = char_model_weight(plate_img, conf=0.3)
                bbox_char = plate_output[0].boxes.xyxy
                cls_char = plate_output[0].boxes.cls
                if len(cls_char) > 0:
                    keys = cls_char.cpu().numpy().astype(int)
                    values = bbox_char[:, 0].cpu().numpy().astype(int)
                    sorted_list = sorted(zip(keys, values), key=lambda x: x[1])
                    char_result = ''.join([charclassnames[k] for k, _ in sorted_list])
                    putText_with_outline(result_img, char_result, (x1, max(30, y1-15)),
                                        cv2.FONT_HERSHEY_TRIPLEX, FONT_SCALE_PLATE, (130, 0, 255), 4, FONT_THICKNESS)

                    city_name = ''
                    if len(char_result) == 8:
                        number_part = int(char_result[6:])
                        letter_part = char_result[2].lower().strip()
                        city_name = find_city_by_plate(letter_part, number_part, city_plateinfo)
                    putText_with_outline(result_img, city_name, (x1, max(70, y1-42)),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.73, (0, 180, 225), 4, FONT_THICKNESS)

    # ---- ACCIDENT DETECTION (WITH THRESHOLD) ----
    results_accident = accident_model_weight(img, conf=0.5)
    for result in results_accident:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > THRESHOLDS["accident"]:
                accident_detected = True
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                putText_with_outline(
                    result_img, f"Accident ({conf:.2f})", (x1, max(35, y1-15)),
                    cv2.FONT_HERSHEY_TRIPLEX, FONT_SCALE_ACC, (0, 0, 255), 6, 3
                )

    # ---- ALERT IF ACCIDENT ----
    if accident_detected and (time.time() - last_alert_time[0]) > ALERT_INTERVAL_SEC:
        play_alert_sound()
        last_alert_time[0] = time.time()

    return result_img

# ================== INPUT & OUTPUT =====================

os.makedirs(PATHS["output"], exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_videoname = f'output_{timestamp}.mp4'
output_imagename = f'output_{timestamp}.jpg'
source = PATHS["source"]

if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    img = cv2.imread(source)
    if img is None:
        print("ERROR >> image not found!")
        exit()
    result_img = process_frame(img)
    cv2.imshow('Detection Result', result_img)
    cv2.imwrite(os.path.join(PATHS["output"], output_imagename), result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(os.path.join(PATHS["output"], output_videoname), fourcc, fps, (frame_width, frame_height))
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        tick = time.time()
        result_img = process_frame(img)
        tock = time.time()
        elapsed_time = tock - tick
        fps_text = "FPS: {:.2f}".format(1/elapsed_time)
        putText_with_outline(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (10, 50, 255), 4, 1)
        cv2.imshow('Detection Result', result_img)
        video_writer.write(result_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# github >> REZAGOONER
