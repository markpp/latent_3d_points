# train a generative adversarial network on a one-dimensional function
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from matplotlib import pyplot
import random
import numpy as np
from tools import pptk_viewer

test_y = np.load("output/val_anno.npy")[:,:3]*100
test_x = np.load("output/val_latent.npy")*100
test_pcs = np.load("output/val_pcs.npy")

train_y = np.load("output/train_anno.npy")[:,:3]*100
train_x = np.load("output/train_latent.npy")*100

num_test, pose_dim = test_y.shape
num_test, latent_dim = test_x.shape
num_train, input_dim = train_x.shape

print("Training GAN with {}->GEN->{} and {}->DIS".format(latent_dim,pose_dim+latent_dim,pose_dim+latent_dim))

# define the standalone discriminator model
def define_discriminator(n_inputs=pose_dim+latent_dim):
	model = Sequential()
	model.add(Dense(32, kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(15, kernel_initializer='he_uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(10, kernel_initializer='he_uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(5, kernel_initializer='he_uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(1, activation='sigmoid'))
	
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim, n_outputs=pose_dim+latent_dim):
	model = Sequential()
	model.add(Dense(32, kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(24, kernel_initializer='he_uniform'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dense(n_outputs))
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def generate_real_samples(n): # input to discriminator
	real_X = []
	for _ in range(n):
		idx = random.randint(0,num_train-1)
		real_X.append(np.concatenate((train_x[idx],train_y[idx]),axis=0))
	return np.array(real_X), ones((n, 1))

def generate_latent_points(latent_dim, n): # input to generator
	X = []
	ids = []
	for _ in range(n):
		idx = random.randint(0,num_test-1)
		X.append(test_x[idx])
		ids.append(idx)
	return np.array(X), np.array(ids)

def generate_fake_samples(generator, latent_dim, n): # input to discriminator
	X, ids = generate_latent_points(latent_dim, n)
	fake_X = generator.predict(X)
	return fake_X, zeros((n, 1)), ids
	#return np.concatenate((fake_X,X),axis=0), zeros((n, 1))

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
	# prepare real samples
	x_real, y_real = generate_real_samples(n)
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake, ids_fake = generate_fake_samples(generator, latent_dim, n)
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print("{} acc_real: {:.2f}, acc_fake: {:.2f}".format(epoch, acc_real, acc_fake))
	# scatter plot real and fake data points
	#pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	#pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	#pyplot.show()
	fake_id = ids_fake[0]
	print("real: {}".format(test_y[fake_id,:3]))
	print("fake: {}".format(x_fake[0,:3]))
	#pptk_viewer.show_points_with_point(test_pcs[fake_id], x_fake[0,:3])

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=100000, n_batch=128, n_eval=1000):
	# determine half the size of one batch, for updating the discriminator
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare real samples
		x_real, y_real = generate_real_samples(half_batch)
		# prepare fake examples
		x_fake, y_fake, _ = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		# evaluate the model every n_eval epochs
		if (i) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)
 
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)


''' 
# generate n real samples with class labels
def generate_real_samples(n):
	# generate inputs in [-0.5, 0.5]
	X1 = rand(n) - 0.5
	# generate outputs X^2
	X2 = X1 * X1
	# stack arrays
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# generate class labels
	y = ones((n, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y
'''