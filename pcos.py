import os 
import cv2
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

#!git clone https://github.com/fidodido12/pcos.git
#!mkdir /content/pcos/data/train/notinfected
#!unzip /content/pcos/data/test.zip             -d /content/pcos/data/
#!unzip /content/pcos/data/train/infected.zip   -d /content/pcos/data/train/
#!unzip /content/pcos/data/train/img_0_4093.zip -d /content/pcos/data/train/notinfected/
#!unzip /content/pcos/data/train/img_0_5988.zip -d /content/pcos/data/train/notinfected/
#!unzip /content/pcos/data/train/img_0_6005.zip -d /content/pcos/data/train/notinfected/

#**1.2. Test-Train Data**
#**Split the dataset**
#**os.walk()-->This function gives the possibility to list the contents of a directory. For example, it is used to find out which files and subdirectories are in the current directory.**
#**glob.glob()-->**It is a module that helps to list files in a specific folder in Python. Searches in subfolders
def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count=0
    # crawls inside folders
    for current_path,dirs,files in os.walk(directory):
        for dr in dirs:
            count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
    return count

train_dir ="data/train"
test_dir="data/test"

#train file image count
train_samples =get_files(train_dir)
#to get tags
num_classes=len(glob.glob(train_dir+"/*")) 
#test file image count
test_samples=get_files(test_dir)
print(num_classes,"Classes")
print(train_samples,"Train images")
print(test_samples,"Test images")
#### **1.3. ImageDataGenerator**
#**ImageDataGenerator**,Data augmentation is used to increase the size of training set and to get more different image. Through Data augmentation we can prevent overfitting ,this refers to randomly changing the images in ways that shouldn’t impact their interpretation, such as horizontal flipping, zooming, and rotating
train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )
test_datagen=ImageDataGenerator(rescale=1./255)
#**flow_from_directory() -->** Another method to read images into TensorFlow environment is to use the .flow_from_directory() method. flow_from_directory is an ImageDataGenerator method. The dataset is read with flow_from_directory without making any changes.
#**Parameters:**
#* **directory:** The path of the target directory. It must contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF formatted images found in each of the subdirectories will be included in the generator.
#* **target_size:** A tuple of integers, (height, width), by default (256,256). All found images will be resized.
#* **batch_size:** The size of the data chunks (default: 32).
#* **shuffle:** Decides whether to shuffle data (default: True). If set to false, it sorts the data in alphanumeric order.
input_shape=(224,224,3)
train_generator =train_datagen.flow_from_directory(train_dir,target_size=(224,224),batch_size=32)
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,target_size=(224,224),batch_size=32)
###***1.4. CNN Model
#A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
model = Sequential()
model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu',name="conv2d_1"))
model.add(MaxPooling2D(pool_size=(3, 3),name="max_pooling2d_1"))
model.add(Conv2D(32, (3, 3),activation='relu',name="conv2d_2"))
model.add(MaxPooling2D(pool_size=(2, 2),name="max_pooling2d_2"))
model.add(Conv2D(64, (3, 3),activation='relu',name="conv2d_3"))
model.add(MaxPooling2D(pool_size=(2, 2),name="max_pooling2d_3"))   
model.add(Flatten(name="flatten_1"))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))          
model.add(Dense(num_classes,activation='sigmoid'))
model.summary()

validation_generator = train_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224), 
        batch_size=32)
#When compiling the model, we provide objective function (loss), optimization method (adam) and accuracy that we will follow.
#Double-click (or enter) to edit

# Load the class labels
class_labels = sorted(os.listdir(train_dir))

# Create a subplot with 2 rows and 5 columns
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

# Display 5 random images from each class
for i, class_label in enumerate(class_labels):
    class_path = os.path.join(train_dir, class_label)
    image_files = random.sample(os.listdir(class_path), 5)
    
    for j, image_file in enumerate(image_files):
        image_path = os.path.join(class_path, image_file)
        img = mpimg.imread(image_path)
        
        # Display the image in the subplot
        axes[i, j].imshow(img)
        axes[i, j].set_title(f"Class: {class_label}")
        axes[i, j].axis('off')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()


####
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
history1 = model.fit(
    train_generator,#egitim verileri
    steps_per_epoch=None,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=None,
    verbose=1,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)],
    shuffle=True
    )

model.save('Model/PCOS_MODEL.h5')
model.save('Model/my_model.keras')
#

print(history1.history.keys())

plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#print(47)
#exit(0)
