import numpy as np 
import pandas as pd 
import os
print(os.listdir("D:/Machine-Learning-Resource/Demo/aerial-cactus-identification/"))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, LeakyReLU, Activation
from keras.layers.normalization import BatchNormalization
from keras.applications.densenet import DenseNet201
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# File paths to data
input_path = 'D:/Machine-Learning-Resource/Demo/'
train_path = input_path + 'train/'
test_path = input_path + 'test/'

# Load data
train_df = pd.read_csv(input_path + 'train.csv')
sample = pd.read_csv(input_path + 'sample_submission.csv')

# Extract ids and labels
train_id = train_df['id']
labels = train_df['class']
test_id = sample['id']

# partition dataset
x_train, x_val, y_train, y_val = train_test_split(train_id, labels, test_size=0.2)

def fetch_images(ids, filepath):
    arr = []
    for img_id in ids:
        img = plt.imread(filepath + img_id)
        arr.append(img)
    
    # normalize pixel values
    arr = np.array(arr).astype('float32')
    arr = arr / 255
    return arr

# Redefine sets to contain images and not ids
x_train = fetch_images(ids=x_train, filepath=train_path)
x_val = fetch_images(ids=x_val, filepath=train_path)
test = fetch_images(ids=test_id, filepath=test_path)

# Get dimensions of each image
img_dim = x_train.shape[1:]   

# plot sample of data
#fig, ax = plt.subplots(nrows=2, ncols=3)
#ax = ax.ravel()
#plt.tight_layout(pad=0.2, h_pad=2)
#
#for i in range(6):
#    ax[i].imshow(x_train[i])
#    ax[i].set_title('has_cactus = {}'.format(y_train.iloc[i]))

    
# create CNN model using DenseNet
batch_size = 32
epochs = 15
steps = x_train.shape[0] // batch_size
# Input dimensions
inputs = Input(shape=img_dim)
# DenseNet
densenet201 = DenseNet201(include_top=False)(inputs)
# Fully connected layer
flat1 = Flatten()(densenet201)
dense1 = Dense(units=256, use_bias=True)(flat1)
batchnorm1 = BatchNormalization()(dense1)
act1 = Activation(activation='relu')(batchnorm1)
drop1 = Dropout(rate=0.5)(act1)
# Output
out = Dense(units=1, activation='sigmoid')(drop1)

# Create Model
model = Model(inputs=inputs, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Fix plateau for learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=2, mode='max')

# image augmentation
img_aug = ImageDataGenerator(rotation_range=20,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    zoom_range=0.2,
                    shear_range=5)
img_aug.fit(x_train)

# plot model architecture
from keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='densenet201_model.png')

# fit model
model.fit_generator(img_aug.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=steps, epochs=epochs, 
                    validation_data=(x_val, y_val), callbacks=[reduce_lr], 
                    verbose=1)
# predict on testing data
test_pred = model.predict(test, verbose=1)
