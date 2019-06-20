# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def create_mlp(in_dim, out_dim, regress=False):
	size_factor = 1
	# define our MLP network
	model = Sequential()
	model.add(Dense(24, input_dim=in_dim, activation="tanh"))
	model.add(Dense(16, activation="tanh"))
	model.add(Dense(12, activation="tanh"))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation="tanh"))

	# check to see if the regression node should be added
	if regress:
		#model.add(Dropout(0.5))
		model.add(Dense(out_dim))

	# return our model
	return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("tanh")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(32)(x)
	x = Activation("tanh")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(16)(x)
	x = Activation("tanh")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(3, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model
