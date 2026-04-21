import cv2
from core.yolo import detect_crime

# -------- VIDEO SOURCE --------
cap = cv2.VideoCapture(0)   # 0 = webcam
# cap = cv2.VideoCapture("test.mp4")  # use video file if needed

# -------- OUTPUT VIDEO (optional) --------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # -------- DETECTION --------
    processed_frame, crime = detect_crime(frame)

    # -------- SHOW --------
    cv2.imshow("Crime Detection System", processed_frame)

    # -------- SAVE VIDEO --------
    out.write(processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()