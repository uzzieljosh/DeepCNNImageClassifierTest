import tensorflow as tf
import os
import cv2
# import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = os.path.join(os.getcwd(), 'data')

# img = cv2.imread(os.path.join(data_dir, 'cat', 'cat.0.jpg'))
# plt.imshow(img)

data = tf.keras.utils.image_dataset_from_directory('data')
# data_iterator = data.as_numpy_iterator()
# batch = data_iterator.next()

# Class 1 = Dog
# Class 0 = Cat
# fig, ax = plt.subplots(ncols=4,figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].set_title(batch[1][idx])

data = data.map(lambda x, y: (x/255, y))

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential()

# Add convulutional layer and a maxpooling layer || first layer needs input shape
model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
model.summary()

logdir = os.path.join(os.getcwd(), 'logs')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

fig= plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss')
plt.legend(loc="upper left")
plt.show()

# from keras.metrics import BinaryAccuracy, Precision, Recall

# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()

# for batch in test.as_numpy_iterator():
#     X, y = batch
#     y_pred = model.predict(X)
#     pre.update_state(y, y_pred)
#     re.update_state(y, y_pred)
#     acc.update_state(y, y_pred)
# print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

img = cv2.imread(os.path.join(data_dir, 'cattest.jpg'))
resize = tf.image.resize(img, (256,256))
accu = model.predict(np.expand_dims(resize/255, 0))
print(accu)
if(accu<0.5):
    print('Cat')
else:
    print('Dog')

img = cv2.imread(os.path.join(data_dir, 'dogtest.jpg'))
resize = tf.image.resize(img, (256,256))
accu = model.predict(np.expand_dims(resize/255, 0))
print(accu)
if(accu<0.5):
    print('Cat')
else:
    print('Dog')

model.save(os.path.join(os.getcwd(), 'dogcatclassifiermodel.h5'))
