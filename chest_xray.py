# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Install required packages for Colab
# !pip install scikit-image imbalanced-learn opencv-python matplotlib seaborn joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import cv2
import os
import random

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150  # Standard size for feature extraction

# Đường dẫn dữ liệu trên máy bạn
DATA_DIR = r'd:\season 4 ver 1\BTL_PPHM\chest_xray'

# Tạo dataset mới với tỷ lệ 80-10-10
import shutil
import random

def create_balanced_dataset():
    """Tạo dataset mới với tỷ lệ 80-10-10"""
    new_dataset_path = r'd:\season 4 ver 1\BTL_PPHM\chest_xray_balanced'
    
    if not os.path.exists(new_dataset_path):
        print("Creating balanced dataset with 80-10-10 split...")
        
        # Tạo thư mục mới
        for split in ['train', 'val', 'test']:
            for cls in ['NORMAL', 'PNEUMONIA']:
                os.makedirs(f'{new_dataset_path}/{split}/{cls}', exist_ok=True)
        
        # Xử lý từng class
        for cls in ['NORMAL', 'PNEUMONIA']:
            all_files = []
            
            # Gộp tất cả ảnh từ train/val/test cũ
            for split in ['train', 'val', 'test']:
                source_folder = f'{DATA_DIR}/{split}/{cls}'
                if os.path.exists(source_folder):
                    files = os.listdir(source_folder)
                    all_files.extend([(file, source_folder) for file in files])
            
            print(f"{cls}: Found {len(all_files)} total images")
            
            # Xáo trộn ngẫu nhiên
            random.shuffle(all_files)
            
            # Chia theo tỷ lệ 80-10-10
            train_files = all_files[:int(len(all_files)*0.8)]
            val_files = all_files[int(len(all_files)*0.8):int(len(all_files)*0.9)]
            test_files = all_files[int(len(all_files)*0.9):]
            
            print(f"{cls}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            
            # Copy files
            for file, source_folder in train_files:
                dest = f'{new_dataset_path}/train/{cls}/{file}'
                shutil.copy(f'{source_folder}/{file}', dest)
            
            for file, source_folder in val_files:
                dest = f'{new_dataset_path}/val/{cls}/{file}'
                shutil.copy(f'{source_folder}/{file}', dest)
            
            for file, source_folder in test_files:
                dest = f'{new_dataset_path}/test/{cls}/{file}'
                shutil.copy(f'{source_folder}/{file}', dest)
        
        print("Balanced dataset created successfully!")
        return new_dataset_path
    else:
        print("Balanced dataset already exists!")
        return new_dataset_path

# Tạo dataset cân bằng
BALANCED_DATA_DIR = create_balanced_dataset()

def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        print(f"Processing {label} images...")
        count = 0
        error_count = 0
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is not None:
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, class_num])
                    count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
        print(f"✓ {label}: {count} images loaded successfully, {error_count} errors")
    return data

def get_balanced_training_data(data_dir):
    data_pneumonia = []
    data_normal = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is not None:
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    if class_num == 0:
                        data_pneumonia.append([resized_arr, class_num])
                    else:
                        data_normal.append([resized_arr, class_num])
            except Exception as e:
                pass
    # Lấy số lượng nhỏ nhất giữa hai lớp
    min_len = min(len(data_pneumonia), len(data_normal))
    data_pneumonia = random.sample(data_pneumonia, min_len)
    data_normal = random.sample(data_normal, min_len)
    data = data_pneumonia + data_normal
    random.shuffle(data)
    print(f"Balanced train set: {len(data_pneumonia)} PNEUMONIA, {len(data_normal)} NORMAL")
    return data

# Sử dụng dataset cân bằng với tỷ lệ 80-10-10
train = get_balanced_training_data(os.path.join(BALANCED_DATA_DIR, 'train'))
test = get_training_data(os.path.join(BALANCED_DATA_DIR, 'test'))
val = get_training_data(os.path.join(BALANCED_DATA_DIR, 'val'))

l = []
for i in train:
    if(i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)        
plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])

x_train, y_train = [], []
x_val, y_val = [], []
x_test, y_test = [], []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Feature extraction functions
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

print("Extracting HOG features...")
hog_features_train = extract_hog_features(x_train)
hog_features_val = extract_hog_features(x_val)
hog_features_test = extract_hog_features(x_test)

print("Extracting LBP features...")
lbp_features_train = extract_lbp_features(x_train)
lbp_features_val = extract_lbp_features(x_val)
lbp_features_test = extract_lbp_features(x_test)

print("Extracting GLCM features...")
glcm_features_train = extract_glcm_features(x_train)
glcm_features_val = extract_glcm_features(x_val)
glcm_features_test = extract_glcm_features(x_test)

print("Extracting color features...")
color_features_train = extract_color_features(x_train)
color_features_val = extract_color_features(x_val)
color_features_test = extract_color_features(x_test)

print("Combining features...")
X_train = np.concatenate([hog_features_train, lbp_features_train, glcm_features_train, color_features_train], axis=1)
X_val = np.concatenate([hog_features_val, lbp_features_val, glcm_features_val, color_features_val], axis=1)
X_test = np.concatenate([hog_features_test, lbp_features_test, glcm_features_test, color_features_test], axis=1)

print(f"Feature shape: {X_train.shape}")

print(f"Number of samples per class: {np.bincount(y_train)}")
print(f"Class distribution: NORMAL={np.bincount(y_train)[1]}, PNEUMONIA={np.bincount(y_train)[0]}")
print(f"Dataset split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
print(f"Train ratio: {len(y_train)/(len(y_train)+len(y_val)+len(y_test)):.1%}")
print(f"Val ratio: {len(y_val)/(len(y_train)+len(y_val)+len(y_test)):.1%}")
print(f"Test ratio: {len(y_test)/(len(y_train)+len(y_val)+len(y_test)):.1%}")

print("Applying SMOTE to balance data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=200)  # Tăng từ 50 lên 200
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# Điều chỉnh class weights mạnh hơn để giảm bias về PNEUMONIA
class_weights = {0: 0.5, 1: 2.0}  # Giảm mạnh weight cho PNEUMONIA, tăng mạnh cho NORMAL

models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=class_weights, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight=class_weights, max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight=class_weights)
}

# Thay thế phần huấn luyện model bằng GridSearchCV cho từng model
param_grids = {
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf'],
        'probability': [True],
        'class_weight': [class_weights]
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': [class_weights]
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'max_iter': [1000],
        'class_weight': [class_weights]
    },
    'Naive Bayes': {},  # Không cần tối ưu hyperparameters
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [class_weights]
    }
}

best_model = None
best_score = 0
results = {}

for name, model in models.items():
    print(f"\nGridSearchCV for {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_pca, y_train_balanced)
    y_val_pred = grid.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    results[name] = val_accuracy
    print(f"{name} Validation Accuracy: {val_accuracy:.3f}")
    print(f"Best params: {grid.best_params_}")
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_model = grid.best_estimator_

print(f"\nBest model: {max(results, key=results.get)} with accuracy: {max(results.values()):.3f}")

model = best_model
y_test_pred = model.predict(X_test_pca)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nFinal Test Accuracy: {test_accuracy:.3f}")
print(f"Best model: {max(results, key=results.get)}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)']))

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
plt.figure(figsize=(10, 10))
sns.heatmap(cm_df, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', 
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.show()

# Thêm ROC-AUC evaluation
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# Tính ROC-AUC
y_test_proba = model.predict_proba(X_test_pca)[:, 1]
auc_score = roc_auc_score(y_test, y_test_proba)
print(f"\nROC-AUC Score: {auc_score:.3f}")

# Threshold tuning để cân bằng precision/recall
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)

# Tìm threshold tối ưu (cân bằng precision và recall)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Optimal F1-score: {f1_scores[optimal_idx]:.3f}")

# Dự đoán với threshold tối ưu
y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)
optimal_accuracy = accuracy_score(y_test, y_test_pred_optimal)
print(f"Accuracy with optimal threshold: {optimal_accuracy:.3f}")

print("\nClassification Report with optimal threshold:")
print(classification_report(y_test, y_test_pred_optimal, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)']))

# Vẽ ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Vẽ Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()

# Lưu model và các đối tượng tiền xử lý vào thư mục hiện tại
joblib.dump(model, 'chest_xray_ml_model.pkl')
joblib.dump(scaler, 'chest_xray_scaler.pkl')
joblib.dump(pca, 'chest_xray_pca.pkl')

print("\nModel and preprocessing objects saved successfully!")
print("Files saved:")
print("- chest_xray_ml_model.pkl (trained model)")
print("- chest_xray_scaler.pkl (feature scaler)")
print("- chest_xray_pca.pkl (PCA transformer)")

def augment_image(img, label):
    """Augmentation khác nhau cho từng class"""
    augmented_imgs = [img]  # Giữ ảnh gốc
    
    if label == 1:  # NORMAL class - tăng cường augmentation
        # Tạo 3 ảnh augmented cho NORMAL
        for _ in range(3):
            aug_img = img.copy()
            
            # Lật ngang (50% chance)
            if random.random() > 0.5:
                aug_img = np.fliplr(aug_img)
            
            # Xoay nhẹ (-15 đến +15 độ)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                # Simple rotation simulation
                if angle > 0:
                    aug_img = np.rot90(aug_img, k=1)
                elif angle < 0:
                    aug_img = np.rot90(aug_img, k=3)
            
            # Thay đổi độ sáng (nhẹ hơn cho NORMAL)
            if random.random() > 0.5:
                factor = 0.9 + 0.2 * random.random()  # 0.9-1.1
                aug_img = np.clip(aug_img * factor, 0, 1)
            
            # Thêm noise nhẹ
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.01, aug_img.shape)
                aug_img = np.clip(aug_img + noise, 0, 1)
            
            augmented_imgs.append(aug_img)
    
    else:  # PNEUMONIA class - augmentation nhẹ hơn
        # Chỉ tạo 1 ảnh augmented cho PNEUMONIA
        aug_img = img.copy()
        
        # Lật ngang (30% chance)
        if random.random() > 0.7:
            aug_img = np.fliplr(aug_img)
        
        # Thay đổi độ sáng (rất nhẹ)
        if random.random() > 0.7:
            factor = 0.95 + 0.1 * random.random()  # 0.95-1.05
            aug_img = np.clip(aug_img * factor, 0, 1)
        
        augmented_imgs.append(aug_img)
    
    return augmented_imgs

# Áp dụng augmentation cho từng class
print("Applying class-specific augmentation...")
x_train_augmented = []
y_train_augmented = []

for i, (img, label) in enumerate(zip(x_train, y_train)):
    # Thêm ảnh gốc
    x_train_augmented.append(img)
    y_train_augmented.append(label)
    
    # Thêm ảnh augmented
    augmented_imgs = augment_image(img, label)
    for aug_img in augmented_imgs[1:]:  # Bỏ qua ảnh gốc
        x_train_augmented.append(aug_img)
        y_train_augmented.append(label)

x_train = np.array(x_train_augmented)
y_train = np.array(y_train_augmented)

print(f"After augmentation - Class distribution: {np.bincount(y_train)}")
print(f"NORMAL: {np.bincount(y_train)[1]}, PNEUMONIA: {np.bincount(y_train)[0]}")

def preprocess_image(img):
    img = (img * 255).astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

x_train = np.array([preprocess_image(img) for img in x_train])
x_val = np.array([preprocess_image(img) for img in x_val])
x_test = np.array([preprocess_image(img) for img in x_test])