import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage import io
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from joblib import dump

from modules.model.circle_detection.hough_transform import extract_hog_features


def train_svm(train_file, kernel='linear', C=1.0, model_save_path='svm_model.joblib', scaler_save_path='scaler.joblib'):
    try:
        train_data = pd.read_csv(train_file)

        required_columns = ['image_path', 'diameter', 'label']
        if not all(column in train_data.columns for column in required_columns):
            raise ValueError("input data must contain the required columns : 'image_path', 'diameter', 'label'.")

        hog_features_list = []
        for image_path in train_data['image_path']:
            hog_features = extract_hog_features(image_path)
            hog_features_list.append(hog_features)
        
        X_hog = np.array(hog_features_list)
        X_train = np.hstack((train_data[['diameter']].values, X_hog))

        y_train = train_data['label']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        svm_model = svm.SVC(kernel=kernel, C=C)
        svm_model.fit(X_train_scaled, y_train)

        dump(svm_model, model_save_path)
        dump(scaler, scaler_save_path)

        return svm_model
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

