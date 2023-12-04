# Import necessary libraries
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert labels to binary classification (bird or not a bird)
y_train = (y_train == 2).astype(int)
y_test = (y_test == 2).astype(int)

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Create a convolutional neural network model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=1,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save the trained model to a file
model.save("bird_model.h5")

# Load the trained model
model = load_model('bird_model.h5')

# Make predictions on test data
predictions = model.predict(x_test, batch_size=32, verbose=1)

# Convert predictions to binary (bird or not a bird) based on a threshold
predictions = predictions > 0.5

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print("True Positives: ", tp)
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)

# Calculate Precision and Recall for each class
report = classification_report(y_test, predictions)
print(report)
