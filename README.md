# CHUAN-DOAN-VIEM-PHOI-XQUANG
Chuẩn Đoán Viêm Phổi Qua Ảnh X-Quang
# 🩺 Chest X-Ray Classification (Pneumonia vs Normal)

Dự án này thực hiện **phân loại ảnh X-quang phổi** thành hai nhãn:  
- **PNEUMONIA**  
- **NORMAL**

---

## 🚀 Các bước thực hiện
- Tiền xử lý ảnh (resize, histogram equalization).
- Tạo dataset cân bằng theo tỷ lệ 80-10-10.
- Data augmentation khác nhau cho từng lớp.
- Trích xuất đặc trưng:  
  - HOG (Histogram of Oriented Gradients)  
  - LBP (Local Binary Pattern)  
  - GLCM (Gray Level Co-occurrence Matrix)  
  - Color Features (Mean, Std, Skewness, Kurtosis)  
- Cân bằng dữ liệu bằng **SMOTE**.
- Giảm chiều dữ liệu bằng **PCA**.
- Huấn luyện nhiều mô hình ML (SVM, Random Forest, Logistic Regression, Naive Bayes, KNN, Decision Tree).
- Đánh giá mô hình bằng:  
  - Accuracy  
  - Confusion Matrix  
  - ROC-AUC  
  - Precision-Recall Curve  

---

## 📂 Cấu trúc dự án
