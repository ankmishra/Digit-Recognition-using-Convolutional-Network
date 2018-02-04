from keras. datasets import mnist 
from keras. models import Sequential  
from keras. layers import Dense  
from keras. layers import Dropout  
from keras. layers import Flatten  
import numpy as np
from matplotlib import pyplot as plt
from keras. layers . convolutional import Conv2D  
from keras. layers . convolutional import MaxPooling2D  
from keras. utils import np_utils 
from keras import backend as K 
K . set_image_dim_ordering ( 'th' )
import cv2
import matplotlib. pyplot as plt 

#Divided the data into subsets of training and testing.
( X_train , y_train ) , ( X_test , y_test ) = mnist. load_data ( )
# Since we are working in gray scale we can
# set the depth to the value 1.
X_train = X_train . reshape ( X_train . shape [ 0 ] , 1 , 28 , 28 ) . astype ( 'float32' )
X_test = X_test . reshape ( X_test . shape [ 0 ] , 1 , 28 , 28 ) . astype ( 'float32' )
# We normalize our data according to the
# gray scale. The floating point values ​​are in the range [0,1], instead of [.255]
X_train = X_train / 255
X_test = X_test / 255
# Converts y_train and y_test, which are class vectors, to a binary class array (one-hot vectors)
y_train = np_utils. to_categorical ( y_train )
y_test = np_utils. to_categorical ( y_test )
# Number of digit types found in MNIST. In this case, the value is 10, corresponding to (0,1,2,3,4,5,6,7,8,9).
num_classes = y_test. shape [ 1 ]


def deeper_cnn_model ( ) :
    model = Sequential ( )
    # Convolution2D will be our input layer. We can observe that it has
    # 30 feature maps with size of 5 × 5 and an activation function of type ReLU. 
    model.add ( Conv2D ( 30 , ( 5 , 5 ) , input_shape = (1,28,28) , activation = 'relu' ) )
    # The MaxPooling2D layer will be our second layer where we will have a sample window of size 2 x 2
    model.add ( MaxPooling2D ( pool_size = ( 2 , 2 ) ) )
    # A new convolutional layer, with 15 feature maps of size 3 × 3, and activation function ReLU 
    model.add ( Conv2D ( 15 , ( 3 , 3 ) , activation = 'relu' ) )
    # A new subsampling with a 2x2 dimension pooling.
    model.add ( MaxPooling2D ( pool_size = ( 2 , 2 ) ) )
    
    # We include a dropout with a 20% probability (you can try other values)
    model.add ( Dropout ( 0.2 ) )
    # We need to convert the output of the convolutional layer, so that it can be used as input to the densely connected layer that is next. 
    # What this does is "flatten / flatten" the structure of the output of the convolutional layers, creating a single long vector of features
    # that will be used by the Fully Connected layer.
    model.add ( Flatten ( ) )
    # Fully connected layer with 128 neurons.
    model.add ( Dense ( 128 , activation = 'relu' ) )
    # Followed by a new fully connected layer with 64 neurons
    model.add ( Dense ( 64 , activation = 'relu' ) )
    
    # Followed by a new fully connected layer with 32 neurons
    model.add ( Dense ( 32 , activation = 'relu' ) )
    # The output layer has the number of neurons compatible with the 
    # number of classes to be obtained. Notice that we are using a softmax activation function,
    model.add ( Dense ( num_classes, activation = 'softmax' , name = 'preds' ) )
    # Configure the entire training process of the neural network
    model.compile ( loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = [ 'accuracy' ] )
    
    return model


model = deeper_cnn_model ( )
model.summary ( )
model.fit ( X_train , y_train, validation_data = ( X_test , y_test ) , epochs = 1 , batch_size = 200 )
scores = model.evaluate ( X_test , y_test, verbose = 0 )
print ( "\ nacc:% .2f %%" % (scores [1] * 100)) 


img_pred = cv2.imread ("number-two.png", 0)
img_pred.shape
# forces the image to have the input dimensions equal to those used in the training data (28x28)
if img_pred.shape!= [28,28]:
    img2 = cv2.resize (img_pred, (28,28))
    img_pred = img2.reshape (28,28, -1)
else:
    img_pred = img_pred.reshape (28,28, -1)
    

# here also we inform the value for the depth = 1, number of rows and columns, which correspond 28x28 of the image.
img_pred = img_pred.reshape (1,1,28,28)

pred = model.predict_classes (img_pred)

pred_proba = model.predict_proba (img_pred)
pred_proba = "% .2f %%"% (pred_proba [0] [pred] * 100)

print (pred [0], "with probability of", pred_proba)
