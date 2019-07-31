import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Dropout, Dense, Flatten, Input
from tensorflow.keras.regularizers import l2

activation_function = "relu" #"tanh"
def create_mlp(in_dim, out_dim, regress=False):
	size_factor = 1
	# define our MLP network
	model = Sequential()
	model.add(Dense(24, input_dim=in_dim, activation=activation_function))
	model.add(Dense(16, activation=activation_function))
	model.add(Dense(12, activation=activation_function))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation=activation_function))

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
		x = Activation(activation_function)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(32)(x)
	x = Activation(activation_function)(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(16)(x)
	x = Activation(activation_function)(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(3, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

def train(input_dim, output_dim, train_x, train_y, test_x, test_y, name):
    #do k-fold https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    model = create_mlp(input_dim, output_dim, regress=True)
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-3, decay=1e-3 / 200))

    # train the model
    print("[INFO] training model...")
    print("train_x {}, train_y {}, test_x {}, test_y {}".format(train_x.shape,train_y.shape,test_x.shape,test_y.shape))
    train_hist = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=300, batch_size=128)

    # serialize model to JSON
    model_json = model.to_json()
    with open("trained_model/{}_nn_model.json".format(name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("trained_model/{}_nn_weights.h5".format(name))

    #model.save('trained_model/nn_model_weights.h5')
    return [(name, train_hist)]

def load(name):
    # load json and create model
    json_file = open('trained_model/{}_nn_model.json'.format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("trained_model/{}_nn_weights.h5".format(name))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
    return model