from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import cv2

data_dir = os.path.join(os.getcwd(), 'data')

model = load_model(os.path.join(os.getcwd(),'dogcatclassifiermodel.h5'))

img = cv2.imread(os.path.join(data_dir, 'dogtest.jpg'))
resize = tf.image.resize(img, (256,256))
accu = model.predict(np.expand_dims(resize/255, 0))
print(accu)
if(accu<0.5):
    print('Cat')
else:
    print('Dog')