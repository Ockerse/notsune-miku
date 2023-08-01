import os
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import imghdr 
from keras.layers import Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.callbacks import TensorBoard 
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, BinaryAccuracy
import cv2
import tensorflow as tf


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

data_dir = 'data' 

image_exts = ['jpeg','jpg', 'bmp', 'png']

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory('data', batch_size=32, shuffle=True)


# Determine the sizes for train, val, and test sets
data_size = len(data)
train_size = int(data_size * 0.7)
val_size = int(data_size * 0.2)
test_size = int(len(data)*0.1)

# Shuffle the entire dataset
data = data.shuffle(buffer_size=data_size, seed=42)

# Split the dataset into train, val, and test
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Normalize the data
train = train.map(lambda x, y: (x / 255, y))
val = val.map(lambda x, y: (x / 255, y))
test = test.map(lambda x, y: (x / 255, y))

# Compute class weights
y_train = np.concatenate([y for _, y in train], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Compute class weights
y_train = np.concatenate([y for _, y in train], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

train_labels = np.concatenate([y for _, y in train], axis=0)

# Print the unique class labels
print(np.unique(train_labels))


model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Adding dropout layer with dropout rate of 0.5
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Adding another dropout layer
model.add(Dense(1, activation='sigmoid'))


learning_rate = 0.001
opt = Adam(learning_rate=learning_rate)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

logdir = 'logs'

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)




# Train the model with early stopping
hist = model.fit(train, epochs=50, validation_data=val, callbacks=[tensorboard_callback], class_weight=class_weights)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print("Precision:", pre.result().numpy())
print("Recall:", re.result().numpy())
print("Binary Accuracy:", acc.result().numpy())

model.save(os.path.join('models','updated_imageclassifier2.h5'))