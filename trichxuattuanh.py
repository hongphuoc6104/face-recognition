import cv2
import os

# Đặt đường dẫn thư mục chứa ảnh gốc và thư mục lưu dataset
raw_images_path = "C:/Users/Phat/Desktop/mushroom/Data"  # Thư mục chứa thư mục con theo ID
dataset_path = "C:/Users/Phat/Desktop/mushroom/dataset"  # Thư mục chứa ảnh đã cắt khuôn mặt

# Tạo thư mục lưu dataset nếu chưa có
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load mô hình phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Duyệt qua từng thư mục ID trong "Data"
for user_id in os.listdir(raw_images_path):
    user_folder = os.path.join(raw_images_path, user_id)
    if not os.path.isdir(user_folder):
        continue  # Bỏ qua nếu không phải thư mục
    
    count = 0  # Đếm số ảnh của từng ID
    
    # Duyệt qua từng ảnh trong thư mục của người dùng
    for file_name in os.listdir(user_folder):
        file_path = os.path.join(user_folder, file_name)
        
        # Đọc ảnh và chuyển sang ảnh xám
        img = cv2.imread(file_path)
        if img is None:
            continue  # Bỏ qua ảnh lỗi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt trong ảnh
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]  # Cắt phần khuôn mặt
            face_resized = cv2.resize(face_img, (100, 100))  # Resize về 100x100
            
            # Lưu ảnh đã lọc theo format "ID_SốThứTự.jpg"
            save_path = os.path.join(dataset_path, f"{user_id}_{count}.jpg")
            cv2.imwrite(save_path, face_resized)
        
        print(f"Đã xử lý {file_name} trong thư mục {user_id}")
    
    print(f"Đã lưu {count} khuôn mặt từ ID {user_id} vào {dataset_path}")

print("Hoàn thành xử lý ảnh!")
