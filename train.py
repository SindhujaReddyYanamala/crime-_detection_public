import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, TimeDistributed, LSTM
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 64
FRAMES = 10
EPOCHS = 12   #  slightly increased
BATCH_SIZE = 8

DATASET_PATH = "dataset"

# ===============================
# LOAD DATA
# ===============================
def load_video_data():
    X, y = [], []
    classes = ["nonfight", "fight"]

    for label, cls in enumerate(classes):
        folder = os.path.join(DATASET_PATH, cls)

        print(f" Loading: {folder}")

        for file in os.listdir(folder):
            if not file.endswith((".avi", ".mp4", ".mov")):
                continue

            path = os.path.join(folder, file)
            cap = cv2.VideoCapture(path)

            frames = []
            count = 0

            while len(frames) < FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                #  skip frames → faster + better generalization
                if count % 3 == 0:
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = frame / 255.0
                    frames.append(frame)

                count += 1

            cap.release()

            if len(frames) == FRAMES:
                X.append(frames)
                y.append(label)

    print(f" Total videos: {len(X)}")
    return np.array(X), to_categorical(y, 2)

# ===============================
# MODELS
# ===============================
def cnn_lstm():
    model = Sequential([
        TimeDistributed(Conv2D(32, (3,3), activation='relu'),
                        input_shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D()),

        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D()),

        TimeDistributed(Flatten()),
        LSTM(64),

        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model


def basic_cnn():
    model = Sequential([
        Conv2D(32,(3,3),activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model

# ===============================
# PLOT + METRICS
# ===============================
def evaluate_and_save(model, history, X_test, y_test, name):

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Accuracy Graph
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.title(f"{name} Accuracy")
    plt.savefig(f"{name}_accuracy.png")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{name}_cm.png")

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred[:,1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"{name} ROC")
    plt.legend()
    plt.savefig(f"{name}_roc.png")

    print(f"\n {name} REPORT:\n")
    print(classification_report(y_true, y_pred_classes))


# ===============================
# TRAIN FUNCTION
# ===============================
def train_model(model, X_train, X_test, y_train, y_test, name):

    print(f"\n Training {name}")

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.save(f"{name}.h5")
    print(f"Saved {name}")

    evaluate_and_save(model, history, X_test, y_test, name)


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    X, y = load_video_data()

    # Split for CNN-LSTM
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train CNN-LSTM (MAIN MODEL)
    train_model(cnn_lstm(), X_train, X_test, y_train, y_test, "CNN_LSTM")

    # Train Basic CNN (for comparison)
    X_img = X[:, 0]
    Xtr, Xte, ytr, yte = train_test_split(
        X_img, y, test_size=0.2, random_state=42
    )

    train_model(basic_cnn(), Xtr, Xte, ytr, yte, "BASIC_CNN")