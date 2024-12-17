import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, MaxPooling2D, SeparableConv2D, \
    GlobalAveragePooling2D, Reshape, multiply, Conv2D, Input, Concatenate
import pandas as pd

matplotlib.use('TkAgg')

print("Device: \n", tf.config.experimental.list_physical_devices())
print(tf.__version__)
print(tf.test.is_built_with_cuda())

# Load your dataset
images = np.load("D:/brainx_fold1.npy")
y = np.load("D:/brainy_fold1.npy")

# Initialize KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2)


# Define the Squeeze and Excite block
def squeeze_excite_block(input_layer, ratio=16):
    filters = input_layer.shape[-1]
    se = GlobalAveragePooling2D()(input_layer)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input_layer, se])
    return se


# Function to define the model
def create_model(input_shape):
    inp = Input(shape=input_shape)

    convs = []
    parrallel_kernels = [11, 9, 7, 5, 3]
    for k in range(len(parrallel_kernels)):
        conv = Conv2D(256, parrallel_kernels[k], padding='same', activation='relu', input_shape=input_shape)(inp)
        convs.append(conv)

    out = Concatenate()(convs)
    conv_model = Model(inputs=inp, outputs=out)

    model = tf.keras.Sequential()
    model.add(conv_model)

    model.add(SeparableConv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(16, (3, 3), padding='same', name='lastconv'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', name='DenseLastPL'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))
    adam = tf.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    return model


# Initialize lists to store metrics
history_all_folds = []

# KFold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(images, y), 1):
    print(f"\nTraining fold {fold}...")

    # Split data into training and validation sets for this fold
    X_train, X_val = images[train_idx], images[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Create the model
    model = create_model(input_shape=images.shape[1:])

    # Define callbacks
    model_checkpoint = ModelCheckpoint(f"my_model/pdcnn_fold{fold}.h5", monitor='val_accuracy', save_best_only=True,
                                       mode='max', verbose=1)
    csv_logger = CSVLogger(f"my_model/pdcnn_history_fold{fold}.csv", separator=',', append=True)
    callbacks = [model_checkpoint, csv_logger]

    # Start time
    start_time = time.time()

    # Train the model
    with tf.device('/GPU:0'):
        history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, y_val),
                            callbacks=callbacks)

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for fold {fold}: {elapsed_time} seconds")

    # Save history of the fold
    history_all_folds.append(history.history)

    # Load the training history from CSV file
    training_history = pd.read_csv(f"my_model/pdcnn_history_fold{fold}.csv")

    # Plot loss and accuracy for each fold
    plt.figure(1)
    plt.plot(training_history['loss'], label=f'Fold {fold} Training Loss')
    plt.plot(training_history['val_loss'], label=f'Fold {fold} Validation Loss')

    plt.figure(2)
    plt.plot(training_history['accuracy'], label=f'Fold {fold} Training Accuracy')
    plt.plot(training_history['val_accuracy'], label=f'Fold {fold} Validation Accuracy')

    # Save loss and accuracy curves for each fold
    plt.figure(1)
    plt.legend()
    plt.title(f'Loss of pdcnn_fold{fold}')
    plt.xlabel('Epoch')
    plt.savefig(f'my_model/Loss_of_pdcnn_fold{fold}.png', dpi=600)

    plt.figure(2)
    plt.legend()
    plt.title(f'Accuracy of pdcnn_fold{fold}')
    plt.xlabel('Epoch')
    plt.savefig(f'my_model/Accuracy_of_pdcnn_fold{fold}.png', dpi=600)

    plt.close('all')
