import cupy as np
import gzip
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def plot_data(train_image, train_label, index):

	plt.title('Label is {label}'.format(label=train_label[index]))
	plt.imshow(train_image[index].reshape(28,28), cmap='gray')
	plt.show()

def init_train_data():
	'''
	RETURN  2D array of train images	
			1D array of train labels
			length of both array (number of train images and labels)
	'''
	train_image = gzip.open('EMNIST_DATASET/emnist-byclass-train-images-idx3-ubyte.gz','r')
	train_label = gzip.open('EMNIST_DATASET/emnist-byclass-train-labels-idx1-ubyte.gz','r')

	train_image_magic_number = int.from_bytes(train_image.read(4), 'big')
	train_image_num_images = int.from_bytes(train_image.read(4), 'big')
	train_image_num_rows = int.from_bytes(train_image.read(4), 'big')
	train_image_num_columns = int.from_bytes(train_image.read(4), 'big')

	image_size = train_image_num_rows

	train_label_magic_number = int.from_bytes(train_label.read(4), 'big')
	train_label_num_images = int.from_bytes(train_label.read(4), 'big')

	train_image_buffer = train_image.read(image_size * image_size * train_image_num_images)
	train_image_data = np.frombuffer(train_image_buffer, np.uint8)
	train_image = train_image_data.reshape(train_image_num_images, image_size * image_size)

	train_label_buffer = train_label.read(train_label_num_images)
	train_label = np.frombuffer(train_label_buffer, np.uint8)

	return train_image.T/255, train_label, train_label_num_images

def init_test_data():
	'''
	RETURN  2D array of test images
			1D array of test labels
			length of both array (number of test images and labels)
	'''
	test_image = gzip.open('EMNIST_DATASET/emnist-byclass-test-images-idx3-ubyte.gz','r')
	test_image = gzip.open('EMNIST_DATASET/emnist-byclass-test-labels-idx1-ubyte.gz','r')
	
	test_image_magic_number = int.from_bytes(test_image.read(4), 'big')
	test_image_num_images = int.from_bytes(test_image.read(4), 'big')
	test_image_num_rows = int.from_bytes(test_image.read(4), 'big')
	test_image_num_columns = int.from_bytes(test_image.read(4), 'big')

	image_size = test_image_num_rows

	test_label_magic_number = int.from_bytes(test_label.read(4), 'big')
	test_label_num_images = int.from_bytes(test_label.read(4), 'big')

	test_image_buffer = test_image.read(image_size * image_size * test_image_num_images)
	test_image_data = np.frombuffer(test_image_buffer, np.uint8)

	test_image = test_image_data.reshape(test_image_num_images, image_size * image_size)

	test_label_buffer = test_label.read(test_label_num_images)
	test_label = np.frombuffer(test_label_buffer, np.uint8)

	return test_images.T/255, test_label, test_image_num_images

'''
Activation Functions
'''
def sigmoid(array):
	return 1/(1 + np.exp(-array))

def ReLU(array):
	return np.maximum(array, 0)

def softmax(array):
	return np.exp(array) / sum(np.exp(array))

def tanh(array):
	return np.tanh(array)

def init_params():
	'''
	This is for 784 -> 392 -> 196 -> 98 -> 62 
	784 and 62 has to stay since its the input and output

	add or remove number of nodes by just changing the numbers
	like 392 to 600 would change the shape into 784 -> 600 -> 196 -> 98 -> 62 

	add or remove number of layers by adding the following codes
		weights{# of currentLayer} = sigmoid(np.random.rand({# of nodes of this layer}, {# of nodes of previous layer}))
		biases{# of currentLayer} = sigmoid(np.random.rand({# of nodes of thi layer}, 1))
	make sure to return the weights by the end of this method
	
	*** I know its easier to pass all of the params using array 
		but then the process gets bottlenecked by the array 
		(Numpy is so fast that default array iteration of python is relatively bottlenecking)

	RETURN	2D array of weights1
			1D array of biases1
			...
			2D array of weightsN
			1D array of biasesN
	'''
	weights1 = np.random.rand(250, 784) *  2 - 1
	biases1 = np.random.rand(250, 1) *  2 - 1

	weights2 = np.random.rand(250, 250) *  2 - 1
	biases2 = np.random.rand(250, 1) *  2 - 1

	weights3 = np.random.rand(250, 250) *  2 - 1
	biases3 = np.random.rand(250, 1) *  2 - 1

	weights4 = np.random.rand(250, 250) *  2 - 1
	biases4 = np.random.rand(250, 1) *  2 - 1

	weights5 = np.random.rand(62, 250) *  2 - 1
	biases5 = np.random.rand(62, 1) *  2 - 1
	
	return weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5

def forward_propagation(image, activation, weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5):
	'''
	image = 2D-array with dimension of (784, # of images)
	activation = activation function that will be used
	weights = weights initialized by init_params()
	biases = biases initialized by init_params()

	This is where you connect all the nodes together from the images to result

	As you add layers from init_params(), make sure to add following codes
		network{# of current network} = np.dot(weight{# of current network}, node{# of previous node}) + biases{# of current network}
		node{# of current node} = activation(network{# of current network})

	*** You can change the activation functions of your own
		However make sure to remember so that you can derive them properly later

	*** Make sure to not touch the network1 (or keep in mind the node0 is going to be the image)

	*** Make sure to not touch the last network and node (they are set to 62 : which is the actual number of final labels) 
		and the actiavtion function here does not contribute to the AI.
		It has to be Softmax for proper prediction to be made.

	RETURN	2D array of network1 (think as nodes before being activated)
			2D array of node1
			...
			2D array of networkN
			2D array of nodeN
	'''
	node0 = image

	network1 = weights1.dot(node0) + biases1
	node1 = sigmoid(network1)

	network2 = weights2.dot(node1) + biases2
	node2 = sigmoid(network2)

	network3 = weights3.dot(node2) + biases3
	node3 = sigmoid(network3)

	network4 = weights4.dot(node3) + biases4
	node4 = sigmoid(network4)

	network5 = weights5.dot(node4) + biases5
	node5 = softmax(network5)

	return network1, node1, network2, node2, network3, node3, network4, node4, network5, node5

def expected_result(label):
	'''
	label = 1D-array with dimension of number of images

	RETURN	2D-array with size of [Number of labels * Number of images] with 0s and 1s
				correct answer is labeled 1 otherwise 0
	'''
	# result = np.zeros(shape=(label.size, label.max() + 1), dtype=int)
	result = np.ndarray(shape=(int(label.size), int(label.max() + 1)))
	result.fill(0)
	result[np.arange(label.size), label] = 1
	result = result.T

	return result

'''
Derivative of Activation Functions
'''
def ReLU_Derivative(array):
	return array > 0

def Sigmoid_derivative(array):
	return np.exp(-array) / (np.exp(-array) + 1) ** 2

def back_propagation(image, label, deriv_activation, weights1, network1, node1, weights2, network2, node2, weights3, network3, node3, weights4, network4, node4, weights5, network5, node5):
	'''
	image = 2D-array with dimension of [784, # of images]
	label = 1D-array with dimension of [1, # of images]
	deriv_activation = derivative of activation function
	weights = weights initialized by init_params()
	networks = networks returned from forward_propagation()
	nodes = nodes returned from forward_propagation()

	This is where you calculate the offset of each weights and biases

	As you add layers from init_params(), make sure to add following codes
		dX# = weights[#+1].T.dot(dX[#+1]) * deriv_activation(network#)
		dW# = 1 / num_images * dX#.dot(node[#-1].T)
		dB# = 1 / num_images * np.sum(dX#)
	
	*** Make sure to not touch the dX4, dW4, dB4, dX1, dW1, dB1

	RETURN	derived weights
			derived biases
	'''

	num_images = label.size
	result = expected_result(label)

	dX5 = node5 - result
	dW5 = 1 / num_images * dX5.dot(node4.T)
	dB5 = 1 / num_images * np.sum(dX5)

	dX4 = weights5.T.dot(dX5) * Sigmoid_derivative(network4)
	dW4 = 1 / num_images * dX4.dot(node3.T)
	dB4 = 1 / num_images * np.sum(dX4)

	dX3 = weights4.T.dot(dX4) * Sigmoid_derivative(network3)
	dW3 = 1 / num_images * dX3.dot(node2.T)
	dB3 = 1 / num_images * np.sum(dX3)

	dX2 = weights3.T.dot(dX3) * Sigmoid_derivative(network2)
	dW2 = 1 / num_images * dX2.dot(node1.T)
	dB2 = 1 / num_images * np.sum(dX2)

	dX1 = weights2.T.dot(dX2) * Sigmoid_derivative(network1)
	dW1 = 1 / num_images * dX1.dot(image.T)
	dB1 = 1 / num_images * np.sum(dX1)

	return dW1, dB1, dW2, dB2, dW3, dB3, dW4, dB4, dW5, dB5

def update_parameters(W1, dW1, B1, dB1, W2, dW2, B2, dB2, W3, dW3, B3, dB3, W4, dW4, B4, dB4, W5, dW5, B5, dB5, learning_curve):
	'''
	weights = weights initialized by init_params()
	biases = biases initialized by init_params()

	This is where you update the parameters using derived offset

	As you add layers from init_params(), make sure to add following codes
		W# = W# - alpha * dW#
		B# = B# - alpha * dB#

	RETURN 	updated weights
			updated biases
	'''

	W1 = W1 - learning_curve * dW1
	B1 = B1 - learning_curve * dB1
	W2 = W2 - learning_curve * dW2
	B2 = B2 - learning_curve * dB2
	W3 = W3 - learning_curve * dW3
	B3 = B3 - learning_curve * dB3
	W4 = W4 - learning_curve * dW4
	B4 = B4 - learning_curve * dB4
	W5 = W5 - learning_curve * dW5
	B5 = B5 - learning_curve * dB5

	return W1, B1, W2, B2, W3, B3, W4, B4, W5, B5

def get_prediction(array):

	return np.argmax(array, 0)#, dtype=np.longlong)

def get_accuracy(prediction, label):

	asdf = np.sum(prediction == label)#, dtype=np.longlong)

	# print(prediction)

	# print(label)

	# print(asdf)

	return asdf / label.size

def gradient_descent(image, label, num_iterations, learning_curve):

	weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5 = init_params()

	for i in range(num_iterations):
		
		network1, node1, network2, node2, network3, node3, network4, node4, network5, node5 = forward_propagation(image, ReLU, weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5)

		dW1, dB1, dW2, dB2, dW3, dB3, dW4, dB4, dW5, dB5 = back_propagation(image, label, ReLU_Derivative, weights1, network1, node1, weights2, network2, node2, weights3, network3, node3, weights4, network4, node4, weights5, network5, node5)

		weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5 = update_parameters(weights1, dW1, biases1, dB1, weights2, dW2, biases2, dB2, weights3, dW3, biases3, dB3, weights4, dW4, biases4, dB4, weights5, dW5, biases5, dB5, learning_curve)

		if i % 10 == 0:
			# print(node5)
			print("Iteration :", i)
			print("Accuracy :", get_accuracy(get_prediction(node5), label)*100, "%")

	return weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5

def train():

	train_image, train_label, train = init_train_data()

	weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5 = gradient_descent(train_image, train_label, 50000, 0.25)

	np.save("EMNIST/W1", weights1)
	np.save("EMNIST/B1", biases1)
	np.save("EMNIST/W2", weights2)
	np.save("EMNIST/B2", biases2)
	np.save("EMNIST/W3", weights3)
	np.save("EMNIST/B3", biases3)
	np.save("EMNIST/W4", weights4)
	np.save("EMNIST/B4", biases4)
	np.save("EMNIST/W5", weights5)
	np.save("EMNIST/B5", biases5)

train()