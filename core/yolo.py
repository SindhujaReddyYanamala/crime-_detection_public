from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
from twilio.rest import Client

# -------- MODELS --------
person_model = YOLO("yolov8n.pt")
weapon_model = YOLO("runs/detect/weapon_detector8/weights/best.pt")

WEAPON_CLASSES = ['knife', 'handgun']

fight_model = load_model("CNN_LSTM.h5")

# -------- GLOBALS --------
fight_counter = 0
weapon_history = []
SEQUENCE_LENGTH = 10
frame_buffer = []
last_saved_time = 0
SAVE_INTERVAL = 5   # seconds

# -------- LOITERING --------
person_positions = {}
LOITER_TIME = 10  # change to 300 for real use

# -------- ALERT CONTROL --------
alert_counter = 0
ALERT_INTERVAL = 10
first_alert_sent = False

# -------- STORAGE --------
os.makedirs("alerts", exist_ok=True)

# -------- TWILIO --------
account_sid = "AC8466bb0feb296e58ef58b914783655fb"
auth_token = "8ced381557e0fb272437866a48d1a77b"
client = Client(account_sid, auth_token)

# -------- ALERT --------
def send_alert(message):
    try:
        client.messages.create(
            body=message,
            from_='whatsapp:+14155238886',
            to='whatsapp:+917305689714'
        )
        print("Alert sent:", message)
    except Exception as e:
        print("Twilio Error:", e)

def controlled_alert(message):
    global alert_counter, first_alert_sent

    alert_counter += 1

    if not first_alert_sent:
        send_alert(message)
        first_alert_sent = True
        alert_counter = 0
        return

    if alert_counter >= ALERT_INTERVAL:
        send_alert(message)
        alert_counter = 0

# -------- PREPROCESS --------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame.astype("float32") / 255.0
    return frame

def format_alert(message_type):
    current_time = time.strftime("%H:%M:%S")

    if message_type == "weapon":
        return f"⚠️ ALERT: Weapon Detected\nTime: {current_time}\nStatus: Suspicious Activity"

    elif message_type == "fight":
        return f"🚨 ALERT: Fight Detected\nTime: {current_time}\nImmediate Attention Required!"

    elif message_type == "loiter":
        return f"⚠️ ALERT: Loitering Detected\nTime: {current_time}\nPerson idle too long"

    elif message_type == "high":
        return f"🚨 HIGH THREAT ALERT 🚨\nWeapon near person!\nTime: {current_time}"

# -------- MAIN --------
def detect_crime(frame):
    global frame_buffer, person_positions, fight_counter, weapon_history

    crime_detected = False
    person_detected = False
    fight_detected = False

    current_time = time.time()

    # -------- PERSON DETECTION --------
    person_results = person_model(frame, imgsz=640, conf=0.3, verbose=False)[0]

    persons = []

    for box in person_results.boxes:
        cls_id = int(box.cls[0])
        label = person_model.names[cls_id]

        if label == "person":
            person_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2, y2))

            center = ((x1 + x2)//2, (y1 + y2)//2)

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, "PERSON", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # -------- LOITERING --------
            key = f"{x1}_{y1}"
            if key not in person_positions:
                person_positions[key] = current_time
            else:
                duration = current_time - person_positions[key]

                if duration > LOITER_TIME:
                    cv2.putText(frame, "LOITERING", (x1,y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                    
                    controlled_alert(format_alert("loiter"))

    # -------- WEAPON DETECTION --------
    weapon_results = weapon_model(frame, imgsz=640, conf=0.3, verbose=False)[0]

    weapon_now = False

    for box in weapon_results.boxes:
        cls_id = int(box.cls[0])
        label = weapon_model.names[cls_id]
        conf = float(box.conf[0])

        if label.lower() in WEAPON_CLASSES and conf > 0.35:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -------- PERSON RELATION --------
            valid = False
            for px1, py1, px2, py2 in persons:
                if px1 < x1 < px2 and py1 < y1 < py2:
                    valid = True

            if valid:
                weapon_now = True

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,f"WEAPON {conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                
                controlled_alert(format_alert("weapon"))

    # -------- TEMPORAL SMOOTHING --------
    weapon_history.append(weapon_now)
    if len(weapon_history) > 5:
        weapon_history.pop(0)

    if sum(weapon_history) >= 3:
        crime_detected = True

    # -------- FIGHT DETECTION --------
    processed = preprocess_frame(frame)
    frame_buffer.append(processed)

    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer.pop(0)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        frames = np.array(frame_buffer)
        frames = np.expand_dims(frames, axis=0)

        prediction = fight_model.predict(frames, verbose=0)[0][0]

        if prediction > 0.85:
            fight_counter += 1
        else:
            fight_counter = 0

        if fight_counter >= 3:
            fight_detected = True

            
            controlled_alert(format_alert("fight"))

    # -------- FINAL STATUS --------
    # -------- FINAL STATUS --------
    global last_saved_time

    if (crime_detected and person_detected) or (fight_detected and person_detected):
        status = "SUSPICIOUS ACTIVITY"
        color = (0,0,255)

        # SAVE ONLY SUSPICIOUS
        if time.time() - last_saved_time > SAVE_INTERVAL:
            filename = f"alerts/suspicious_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            last_saved_time = time.time()

            controlled_alert(format_alert("high"))

    elif person_detected:
        status = "PEACEFUL"
        color = (0,255,0)

    else:
        status = "NO ACTIVITY"
        color = (255,255,255)

    cv2.putText(frame, status, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame, crime_detected or fight_detected