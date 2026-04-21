import cv2

cap = cv2.VideoCapture("dataset/fight/fi494_xvid.avi")

ret, frame = cap.read()

if not ret:
    print("❌ Cannot read video")
else:
    print("✅ Video working")

cap.release()