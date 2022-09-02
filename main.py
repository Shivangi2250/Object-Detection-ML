import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import numpy as np
import pandas as pd
filename='Desktop/home//anothercat.jpg'


# from IPython.display import Image
# Image(filename='Desktop/home//anothercat.jpg',width=224,height=224)
# from tensorflow.keras.preprocessing import image
# img = image.load_img(filename,target_size=(224,224))
# import matplotlib.pyplot as plt
# plt.imshow(img)
# mobile = tf.keras.applications.mobilenet.MobileNet()

mobile=tf.keras.applications.mobilenet_v2.MobileNetV2()

from tensorflow.keras.preprocessing import image
img = image.load_img(filename,target_size=(224,224))
plt.imshow(img)

resized_img=image.img_to_array(img)
final_image=np.expand_dims(resized_img,axis=0)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)
final_image.shape

predictions=mobile.predict(final_image)

# print(predictions)

# from tensorflow.keras.applications import imagenet_utils

# results = imagenet_utils.decode_predictions(predictions)

# print(results)

# plt.imshow(img)

predictions=mobile.predict(final_image)

results = imagenet_utils.decode_predictions(predictions)

data = pd.DataFrame(results)
objects = data.to_dict()
# print(y)
for i in objects:
    print(objects[i][0][1])
    i += 1
