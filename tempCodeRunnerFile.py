import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import imblearn
import joblib


df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2021.csv')

df.head()

df.info()
df.describe()

df.isnull().sum()

# plot the values of the target variable
sns.countplot(x='Diabetes_binary', data=df, palette='hls')
plt.show()
# get the count of the target variable
print(df['Diabetes_binary'].value_counts())
print(df['Diabetes_binary'].value_counts(normalize=True))

# plot correlation matrix
corr = df.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()

def plot_loss_accuracy(history, learning_rate, regularization_param):
    history_df = pd.DataFrame(history.history)
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history_df) + 1), history_df['loss'], label='Training loss')  # Add 1 to x values
    plt.plot(range(1, len(history_df) + 1), history_df['val_loss'], label='Validation loss')  # Add 1 to x values
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, f'Learning rate: {learning_rate}\nRegularization: {regularization_param}', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history_df) + 1), history_df['accuracy'], label='Training accuracy')
    plt.plot(range(1, len(history_df) + 1), history_df['val_accuracy'], label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, f'Learning rate: {learning_rate}\nRegularization: {regularization_param}', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

# train valid split
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.1, random_state=42)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
# normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# randomly oversample the data
def random_oversampler(X_train, y_train):
    ros = imblearn.over_sampling.RandomOverSampler(random_state=42)
    X_oversampled, y_oversampled = ros.fit_resample(X_train, y_train)
    return X_oversampled, y_oversampled

def get_classification_report(y_pred_raw, y_valid, threshold=0.5):
    y_pred = np.where(y_pred_raw >= threshold, 1, 0)
    print(classification_report(y_valid, y_pred))
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

def build_model(X_train, y_train, X_valid, y_valid,lr=0.001, regularize=0, epochs=20, batch_size=128, callback=[], verbose=1, class_weights=None, threshold=0.5):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    reg = tf.keras.regularizers.l2(regularize)
    # initializer = tf.keras.initializers.HeNormal(seed=42)
    model.add(tfl.Dense(9, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=reg))
    model.add(tfl.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],)
    if class_weights == None:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose = verbose, callbacks=callback)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose = verbose, callbacks=callback, class_weight=class_weights)
    print(model.evaluate(X_valid, y_valid))
    plot_loss_accuracy(history, lr, regularize)
    y_pred_raw = model.predict(X_valid)
    get_classification_report(y_pred_raw, y_valid, threshold)
    return history, model

X_oversampled, y_oversampled = random_oversampler(X_train, y_train)

history_oversamped, model_oversampled = build_model(X_oversampled, y_oversampled, X_valid, y_valid, lr=0.00019, regularize=0, epochs=40, batch_size=1024)

"""Around 70% and 80% recall on negative and positive target variable respectively"""

model_oversampled.save("diabetes_model.keras")
joblib.dump(scaler, "scaler.pkl")