import joblib
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

loaded_model = joblib.load("custom_random_forest_model.pkl")

df = pd.read_pickle("../../data/interim/trainingRealdata.data_features_real.pkl")
df_train = df.drop(["participant", "category", "set"], axis=1)
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_feature = ["acc_r", "gyr_r"]
pca_feature = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]


feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_feature + pca_feature))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))


# Sử dụng mô hình để dự đoán dữ liệu mới
new_data = pd.read_pickle("../../data/interim/2504.data_features_real.pkl")
participant_df = df.drop(["set", "category"], axis=1)
new_df = new_data.drop(["set", "category"], axis=1)
X_train_1 = participant_df.drop("label", axis=1)
y_train_1 = participant_df["label"]
X_test_1 = new_df.drop("label", axis=1)
y_test_1 = new_df["label"]

X_train_1 = X_train_1.drop(["participant"], axis=1)
X_test_1 = X_test_1.drop(["participant"], axis=1)


# Huấn luyện và dự đoán mô hình Random Forest
(
    pred_train_y,
    pred_test_y,
    class_train_prob_y_1,
    class_test_prob_y_1,
) = loaded_model.random_forest(
    X_train_1[feature_set_4], y_train_1, X_test_1[feature_set_4], gridsearch=True
)

# Tính toán độ chính xác trên tập huấn luyện và tập kiểm tra
train_accuracy = accuracy_score(y_train_1, pred_train_y)
test_accuracy = accuracy_score(y_test_1, pred_test_y)

print("Accuracy trên tập huấn luyện:", train_accuracy)
print("Accuracy trên tập kiểm tra:", test_accuracy)
report = classification_report(y_test_1, pred_test_y)

print(report)
