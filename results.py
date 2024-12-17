import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy
import time
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import tensorflow as tf

import matplotlib


matplotlib.use('TkAgg')


###Load numpy X and y
images = np.load("D:/brainX_fold1.npy")
y = np.load("D:/brainy_fold1.npy")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)

# Load the pre-trained model
model = tf.keras.models.load_model("my_model/pdcnn_fold1.h5")
model.summary()
####Load Feature Extractor Layer
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('DenseLastPL').output)
intermediate_layer_model.summary()
# Get the number of layers
num_layers = len(intermediate_layer_model.layers)

print("Number of layers in the CNN model:", num_layers)

# Perform feature extraction
feature_engg_data1 = intermediate_layer_model.predict(images)
feature_engg_data1 = pd.DataFrame(feature_engg_data1)

# Standardize the feature-engineered data
x1 = feature_engg_data1.loc[:, feature_engg_data1.columns].values
x1 = StandardScaler().fit_transform(x1)

# Encode the labels
y1 = tf.keras.utils.to_categorical(y, num_classes=4)

# Single train-test split
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.1, random_state=2)


# Pseudo-inverse ELM
input_size = X_train.shape[1]
hidden_size = 1500

input_weights = np.random.normal(size=[input_size, hidden_size])
biases = np.random.normal(size=[hidden_size])


def relu(x):
    return np.maximum(x, 0, x)


def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H


output_weights = np.dot(scipy.linalg.pinv(hidden_nodes(X_train)), y_train)


def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

start_time_pseudo_inverse = time.time()
prediction = predict(X_test)
end_time_pseudo_inverse = time.time()

# L1 Regularized ELM
input_size = X_train.shape[1]
hidden_size = 1500

input_weights = np.random.normal(size=[input_size, hidden_size])
biases = np.random.normal(size=[hidden_size])

H_train = hidden_nodes(X_train)
alpha_l1 = 0.01  # L1 regularization parameter
output_weights_l1 = np.linalg.lstsq(H_train.T @ H_train + alpha_l1 * np.eye(hidden_size), H_train.T @ y_train, rcond=None)[0]

def predict_l1(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights_l1)
    return out



start_time_l1_regularized = time.time()
prediction_l1 = predict_l1(X_test)
end_time_l1_regularized = time.time()


# Convert probabilities to class predictions
predicted_classes_pseudo_inverse = np.argmax(prediction, axis=1)
predicted_classes_l1_regularized = np.argmax(prediction_l1, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("Results Pseudo-Inverse ELM-----------------------------------------------------")
print(confusion_matrix(true_classes, predicted_classes_pseudo_inverse))
print(classification_report(true_classes, predicted_classes_pseudo_inverse))
print(metrics.accuracy_score(true_classes, predicted_classes_pseudo_inverse))

print("Results L1 Regularized ELM-----------------------------------------------------")
print(confusion_matrix(true_classes, predicted_classes_l1_regularized))
print(classification_report(true_classes, predicted_classes_l1_regularized))
print(metrics.accuracy_score(true_classes, predicted_classes_l1_regularized))


# Print testing times
print("Testing Time for Pseudo-Inverse ELM: {:.4f} seconds".format(end_time_pseudo_inverse - start_time_pseudo_inverse))
print("Testing Time for L1 Regularized ELM: {:.4f} seconds".format(end_time_l1_regularized - start_time_l1_regularized))


#######ROC#############

############### ROC Curves for Pseudo-Inverse ELM
plt.figure()
lw = 1
fpr_pseudo_inverse, tpr_pseudo_inverse, roc_auc_pseudo_inverse = dict(), dict(), dict()
for i in range(4):
    fpr_pseudo_inverse[i], tpr_pseudo_inverse[i], _ = roc_curve(y_test[:, i], prediction[:, i])
    roc_auc_pseudo_inverse[i] = auc(fpr_pseudo_inverse[i], tpr_pseudo_inverse[i])

fpr_pseudo_inverse["micro"], tpr_pseudo_inverse["micro"], _ = roc_curve(y_test.ravel(), prediction.ravel())
roc_auc_pseudo_inverse["micro"] = auc(fpr_pseudo_inverse["micro"], tpr_pseudo_inverse["micro"])

colors = cycle(['red', 'blue', 'green', 'yellow'])
for i, color in zip(range(4), colors):
    plt.plot(fpr_pseudo_inverse[i], tpr_pseudo_inverse[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc_pseudo_inverse[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Pseudo-Inverse ELM')
plt.legend(loc="lower right")
plt.show()
plt.savefig('Data/Save file/ROC for Pseudo-Inverse ELM')

c = roc_auc_score(y_test, prediction, multi_class='ovo')
print("Pseudo-Inverse ELM AUC:", c)


########################## ROC Curves for L1 Regularized ELM
plt.figure()
lw = 1
fpr_l1, tpr_l1, roc_auc_l1 = dict(), dict(), dict()
for i in range(4):
    fpr_l1[i], tpr_l1[i], _ = roc_curve(y_test[:, i], prediction_l1[:, i])
    roc_auc_l1[i] = auc(fpr_l1[i], tpr_l1[i])

fpr_l1["micro"], tpr_l1["micro"], _ = roc_curve(y_test.ravel(), prediction_l1.ravel())
roc_auc_l1["micro"] = auc(fpr_l1["micro"], tpr_l1["micro"])

colors = cycle(['red', 'blue', 'green', 'yellow'])
for i, color in zip(range(4), colors):
    plt.plot(fpr_l1[i], tpr_l1[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'.format(i, roc_auc_l1[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for L1 Regularized ELM')
plt.legend(loc="lower right")
plt.show()
plt.savefig('Data/Save file/ROC for L1 Regularized ELM')

c = roc_auc_score(y_test, prediction_l1, multi_class='ovo')
print("L1 Regularized AUC:", c)
