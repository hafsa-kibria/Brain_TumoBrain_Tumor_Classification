import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shap
import tkinter as tk
from tkinter import filedialog


images = np.load("D:/brainx_fold1.npy")
y = np.load("D:/brainy_fold1.npy")


X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)

# model = tf.keras.models.load_model('E:/Research_Papers/Nahid sir/Waste 2/Model/LPCNNP_1st_Stage.h5')
model = tf.keras.models.load_model('my_model/pdcnn_fold1.h5')
# model = tf.keras.models.load_model('E:/Research_Papers/Nahid sir/Waste 2/Model/LPCNNP_3rd_Stage.h5')


image_path="path of image"
img = cv2.imread(image_path)
img_size = 124  # Same as used during training
img = cv2.resize(img, (img_size, img_size))
img = img / 255.0  # Normalize the pixel values to the range [0, 1]

img = np.expand_dims(img, axis=0)
prediction = model.predict(img)

rounded_labels = np.argmax(prediction, axis=1)
print(rounded_labels)




def SHAP_issue():
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough

SHAP_issue()
background = X_train[np.random.choice(X_train.shape[0], 5, replace=False)]
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(img)

shap.image_plot(shap_values, img, show = True)

