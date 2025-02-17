import cv2
import os

# Đặt đường dẫn đúng
dataset_path = "C:/Users/Phat/Desktop/mushroom/dataset"

# Tạo thư mục dataset nếu chưa có
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load mô hình phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
user_id = input("Nhập ID của bạn: ")

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        file_path = os.path.join(dataset_path, f"{user_id}_{count}.jpg")  # Định dạng "ID_SốThứTự.jpg"
        cv2.imwrite(file_path, gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Đã lưu {count} ảnh vào {dataset_path}")
