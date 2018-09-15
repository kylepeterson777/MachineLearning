import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

def read_img():
	path = os.getcwd() 
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	dirs = os.listdir(path+'/Images/')
	label = 0
	for i in dirs:
		n = 0
		count = 0
		for pic in glob.glob(path+'/Images/'+i+'/*.tif'):
			im = Image.open(pic)
			im = np.array(im)
			if((im.shape[0]==256) and (im.shape[1] ==256)): #get only 90 data
				r = im[:,:,0]
				g = im[:,:,1]
				b = im[:,:,2]
				if(n<50): # 50 data in beginning set as test data
					x_test.append([r,g,b])
					y_test.append([label])
				else: #remaining data set as training data
					x_train.append([r,g,b])
					y_train.append([label])
				n = n + 1
				count = count + 1
		#print(count)
		label = label + 1
	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

img_rows = 256
img_cols = 256
num_class = 21
x_train,y_train,x_test,y_test = read_img()

x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols*3)
x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols*3)

input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 21)
y_test = keras.utils.to_categorical(y_test, 21)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Dense(units=1064,activation="relu",input_shape=(img_rows*img_cols*3,)))
model.add(Dense(units=21,activation="softmax"))

model.compile(optimizer=SGD(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=64,epochs=100,verbose=1)
#model.load_weights("mnist-model.h5")
#model.save("mnistmodel.h5")

accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("Accuracy: ",accuracy[1])


# The predict_classes function outputs the highest probability class# The pr 
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(x_test)

# Check which items we got right / wrong
y_test_ori = np.argmax(y_test,1)
correct_indices = np.nonzero(predicted_classes == y_test_ori)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test_ori)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(img_rows,img_cols,3), interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test_ori[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(img_rows,img_cols,3), interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test_ori[incorrect]))

