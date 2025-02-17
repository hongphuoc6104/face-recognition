import cv2
import numpy as np
import joblib

# Nạp mô hình KNN đã huấn luyện
model_path = "C:/Users/Phat/Desktop/mushroom/knn_model.pkl"
knn = joblib.load(model_path)

# Load bộ nhận diện khuôn mặt Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_resized = cv2.resize(gray[y:y+h, x:x+w], (100, 100)).flatten().reshape(1, -1)
        label = knn.predict(face_resized)[0]  # Dự đoán ID

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition KNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
