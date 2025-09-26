import streamlit as st
import numpy as np
import cv2
import joblib
import os
from sklearn.svm import SVC

st.title("Chẩn đoán viêm phổi từ ảnh X-quang")

MODEL_PATH = 'chest_xray_ml_model.pkl'
SCALER_PATH = 'chest_xray_scaler.pkl'
PCA_PATH = 'chest_xray_pca.pkl'

# Hướng dẫn lưu model nếu chưa có file .pkl
if not os.path.exists(MODEL_PATH):
    st.warning("Bạn cần chạy file chest_xray.py trước để tạo model!")
    st.stop()

# Load model and preprocessing objects
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

# Feature extraction functions (same as in training)
def extract_hog_features(images):
    from skimage.feature import hog
    features = []
    for img in images:
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_features)
    return np.array(features)

def extract_lbp_features(images):
    from skimage.feature import local_binary_pattern
    features = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        features.append(hist)
    return np.array(features)

def extract_glcm_features(images):
    from skimage.feature import graycomatrix, graycoprops
    features = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0, 45, 90, 135], 
                           levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        features.append([contrast, dissimilarity, homogeneity, energy])
    return np.array(features)

def extract_color_features(images):
    features = []
    for img in images:
        mean_val = np.mean(img)
        std_val = np.std(img)
        skewness = np.mean(((img - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((img - mean_val) / std_val) ** 4)
        features.append([mean_val, std_val, skewness, kurtosis])
    return np.array(features)

uploaded_file = st.file_uploader("Chọn ảnh X-quang", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Read as grayscale for ML
    if img is not None:
        # Preprocess image
        img = cv2.resize(img, (150, 150))  # ML input size
        img = img / 255.0
        
        # Display image
        st.image(img, caption="Ảnh X-quang", use_column_width=True)
        
        # Extract features
        hog_features = extract_hog_features([img])
        lbp_features = extract_lbp_features([img])
        glcm_features = extract_glcm_features([img])
        color_features = extract_color_features([img])
        
        # Combine features
        features = np.concatenate([hog_features, lbp_features, glcm_features, color_features], axis=1)
        
        # Preprocess features
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Predict với threshold tối ưu
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(features_pca)
            pneumonia_prob = pred_proba[0][0]
            normal_prob = pred_proba[0][1]
            
            # Sử dụng threshold tối ưu (0.5 -> 0.6 để giảm bias PNEUMONIA)
            optimal_threshold = 0.6
            pred_optimal = 0 if pneumonia_prob >= optimal_threshold else 1
            label = "PNEUMONIA" if pred_optimal == 0 else "NORMAL"
            confidence = max(pneumonia_prob, normal_prob)
        else:
            # Fallback cho model không có predict_proba
            pred = model.predict(features_pca)
            label = "PNEUMONIA" if pred[0] == 0 else "NORMAL"
            confidence = 0.8
            pneumonia_prob = 0.8 if pred[0] == 0 else 0.2
            normal_prob = 0.2 if pred[0] == 0 else 0.8
        
        # Hiển thị kết quả với styling
        st.markdown("---")
        st.markdown("### 🎯 Kết quả dự đoán")
        
        # Tạo 2 cột để hiển thị
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Chẩn đoán:** {label}")
            st.markdown(f"**Độ tin cậy:** {confidence:.2f}")
        
        with col2:
            st.markdown(f"**Xác suất PNEUMONIA:** {pneumonia_prob:.2f}")
            st.markdown(f"**Xác suất NORMAL:** {normal_prob:.2f}")
        
        # Hiển thị thanh progress
        st.markdown("### 📊 Xác suất dự đoán")
        st.progress(pneumonia_prob)
        st.caption(f"PNEUMONIA: {pneumonia_prob:.1%}")
        
        # Thêm cảnh báo nếu độ tin cậy thấp
        if confidence < 0.7:
            st.warning("⚠️ Độ tin cậy thấp. Kết quả có thể không chính xác.")
        elif confidence > 0.9:
            st.success("✅ Độ tin cậy cao. Kết quả đáng tin cậy.")
    else:
        st.error("Không đọc được ảnh. Hãy thử lại với file khác.")
