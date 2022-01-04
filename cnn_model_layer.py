#####################
#
# Contains CNN Model
#
####################


### Build The Network ##
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import concatenate
#from tf.keras.layers import LayerNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import initializers


## CNN NETWORK ##
def make_network(X_DC,X_IC,num_labels,DC_drop_value,IC_drop_value,connected_drop_value):
	
	# DEEP CORE #
	#print("Train Data DC", X_DC.shape)
	strings = X_DC.shape[1]
	dom_per_string = X_DC.shape[2]
	dom_variables = X_DC.shape[3]

	# Conv DC + batch normalization, later dropout and maxpooling
	input_DC = Input(shape=(strings, dom_per_string, dom_variables))

	conv1_DC = Conv2D(100,kernel_size=(strings,5),padding='same',activation='tanh')(input_DC)
	#batch1_DC = LayerNormalization(axis=[0,1])(conv1_DC)
	pool1_DC = MaxPooling2D(pool_size=(1,2))(conv1_DC)
	drop1_DC = Dropout(DC_drop_value)(pool1_DC)

	conv2_DC = Conv2D(100,kernel_size=(strings,7),padding='same',activation='relu')(drop1_DC)
	#batch2_DC = LayerNormalization()(conv2_DC)
	drop2_DC = Dropout(DC_drop_value)(conv2_DC)

	conv3_DC = Conv2D(100,kernel_size=(strings,7),padding='same',activation='relu')(drop2_DC)
	#batch3_DC = LayerNormalization()(conv3_DC)
	drop3_DC = Dropout(DC_drop_value)(conv3_DC)

	conv4_DC = Conv2D(100,kernel_size=(strings,3),padding='valid',activation='relu')(drop3_DC)
	#batch4_DC = LayerNormalization()(conv4_DC)
	pool4_DC = MaxPooling2D(pool_size=(1,2))(conv4_DC)
	drop4_DC = Dropout(DC_drop_value)(pool4_DC)

	conv5_DC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop4_DC)
	#batch5_DC = LayerNormalization()(conv5_DC)
	drop5_DC = Dropout(DC_drop_value)(conv5_DC)

	conv6_DC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop5_DC)
	#batch6_DC = LayerNormalization()(conv6_DC)
	drop6_DC = Dropout(DC_drop_value)(conv6_DC)

	conv7_DC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop6_DC)
	#batch7_DC = LayerNormalization()(conv7_DC)
	drop7_DC = Dropout(DC_drop_value)(conv7_DC)

	conv8_DC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop7_DC)
	#batch8_DC = LayerNormalization()(conv8_DC)
	drop8_DC = Dropout(DC_drop_value)(conv8_DC)

	flat_DC = Flatten()(drop8_DC)


	# ICECUBE NEAR DEEPCORE #
	#print("Train Data IC", X_IC.shape)
	strings_IC = X_IC.shape[1]
	dom_per_string_IC = X_IC.shape[2]
	dom_variables_IC = X_IC.shape[3]

	# Conv DC + batch normalization, later dropout and maxpooling
	input_IC = Input(shape=(strings_IC, dom_per_string_IC, dom_variables_IC))

	conv1_IC = Conv2D(100,kernel_size=(strings_IC,5),padding='same',activation='tanh')(input_IC)
	#batch1_IC = LayerNormalization()(conv1_IC)
	pool1_IC = MaxPooling2D(pool_size=(1,2))(conv1_IC)
	drop1_IC = Dropout(IC_drop_value)(pool1_IC)

	conv2_IC = Conv2D(100,kernel_size=(strings_IC,7),padding='same',activation='relu')(drop1_IC)
	#batch2_IC = LayerNormalization()(conv2_IC)
	drop2_IC = Dropout(IC_drop_value)(conv2_IC)

	conv3_IC = Conv2D(100,kernel_size=(strings_IC,7),padding='same',activation='relu')(drop2_IC)
	#batch3_IC = LayerNormalization()(conv3_IC)
	drop3_IC = Dropout(IC_drop_value)(conv3_IC)

	conv4_IC = Conv2D(100,kernel_size=(strings_IC,3),padding='valid',activation='relu')(drop3_IC)
	#batch4_IC = LayerNormalization()(conv4_IC)
	pool4_IC = MaxPooling2D(pool_size=(1,2))(conv4_IC)
	drop4_IC = Dropout(IC_drop_value)(pool4_IC)

	conv5_IC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop4_IC)
	#batch5_IC = LayerNormalization()(conv5_IC)
	drop5_IC = Dropout(IC_drop_value)(conv5_IC)

	conv6_IC = Conv2D(100,kernel_size=(1,7),padding='same',activation='relu')(drop5_IC)
	#batch6_IC = LayerNormalization()(conv6_IC)
	drop6_IC = Dropout(IC_drop_value)(conv6_IC)

	conv7_IC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop6_IC)
	#batch7_IC = LayerNormalization()(conv7_IC)
	drop7_IC = Dropout(IC_drop_value)(conv7_IC)

	conv8_IC = Conv2D(100,kernel_size=(1,1),padding='same',activation='relu')(drop7_IC)
	#batch8_IC = LayerNormalization()(conv8_IC)
	drop8_IC = Dropout(IC_drop_value)(conv8_IC)

	flat_IC = Flatten()(drop8_IC)


	# PUT TOGETHER #
	concatted = concatenate([flat_DC, flat_IC])

	full1 = Dense(300,activation='relu')(concatted)
	#batch1_full = LayerNormalization()(full1)
	dropf = Dropout(connected_drop_value)(conv1_full)

	output = Dense(num_labels,activation='linear')(dropf)
	model_DC = Model(inputs=[input_DC,input_IC],outputs=output)

	return model_DC
