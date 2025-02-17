import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Đọc ảnh từ dataset
dataset_path = "C:/Users/Phat/Desktop/mushroom/dataset"
image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
faces = []
labels = []

for image_path in image_paths:
    filename = os.path.basename(image_path)
    parts = filename.split("_")  # Tách chuỗi theo dấu "_"
    if len(parts) < 2:
        continue  # Bỏ qua các ảnh không đúng định dạng
    
    try:
        label = int(parts[0])  # Lấy ID từ tên file
    except ValueError:
        continue  # Bỏ qua nếu không thể chuyển đổi thành số nguyên
    
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        continue  # Bỏ qua ảnh lỗi
    
    gray_img_resized = cv2.resize(gray_img, (100, 100))  # Resize ảnh về 100x100
    faces.append(gray_img_resized.flatten())  # Chuyển ảnh thành vector 1D
    labels.append(label)

faces = np.array(faces)
labels = np.array(labels)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Huấn luyện với KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = knn.predict(X_test)

# Hiển thị độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình KNN: {accuracy:.2f}")

# Lưu mô hình KNN
model_path = "C:/Users/Phat/Desktop/mushroom/knn_model.pkl"
joblib.dump(knn, model_path)
print(f"Đã lưu mô hình KNN vào {model_path}")

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print("Ma trận nhầm lẫn:")
print(cm)
print(f"Số ảnh hợp lệ: {len(faces)}")

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(labels), yticklabels=set(labels))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - KNN Face Recognition")
plt.show()
