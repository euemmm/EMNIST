import numpy as np
import gzip
# import pandas as pd
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
import cv2
# import time

def plot_data(train_image, train_label, index):

	plt.title('Label is {label}'.format(label=train_label[index]))
	plt.imshow(train_image[index].reshape(28,28).T, cmap='gray')
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
	test_label = gzip.open('EMNIST_DATASET/emnist-byclass-test-labels-idx1-ubyte.gz','r')
	
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

	return test_image.T/255, test_label, test_image_num_images


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
'''
---------------------
'''


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
	weights1 = np.random.rand(84, 784) *  2 - 1
	biases1 = np.random.rand(84, 1) *  2 - 1

	weights2 = np.random.rand(83, 84) *  2 - 1
	biases2 = np.random.rand(83, 1) *  2 - 1

	weights3 = np.random.rand(81, 83) *  2 - 1
	biases3 = np.random.rand(81, 1) *  2 - 1

	weights4 = np.random.rand(80, 81) *  2 - 1
	biases4 = np.random.rand(80, 1) *  2 - 1

	weights5 = np.random.rand(62, 80) *  2 - 1
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
	node2 = ReLU(network2)

	network3 = weights3.dot(node2) + biases3
	node3 = ReLU(network3)

	network4 = weights4.dot(node3) + biases4
	node4 = sigmoid(network4)

	network5 = weights5.dot(node4) + biases5
	node5 = softmax(network5) #THIS LAST NODE's ACTIVATION FUNCTION NEEDS TO BE SOFTMAX

	return network1, node1, network2, node2, network3, node3, network4, node4, network5, node5

def expected_result(label):
	'''
	label = 1D-array with dimension of number of images

	RETURN	2D-array with size of [Number of labels * Number of images] with 0s and 1s
				correct answer is labeled 1 otherwise 0
	'''
	# result = np.zeros(shape=(label.size, label.max() + 1), dtype=int)
	result = np.zeros((label.size, label.max() + 1))
	# result.fill(0)
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
'''
-----------------------------------
'''


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

	dX3 = weights4.T.dot(dX4) * ReLU_Derivative(network3)
	dW3 = 1 / num_images * dX3.dot(node2.T)
	dB3 = 1 / num_images * np.sum(dX3)

	dX2 = weights3.T.dot(dX3) * ReLU_Derivative(network2)
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
	'''
	This is basically where the actual training happens
	Make sure the weights, biases, networks, nodes, derivatives of each are updated according to your decision of model
	'''

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
	'''
	running this will start the training
	once the training is done, it will save the resulting parameters
	move them into folders, change the folder name so its distinguishable, add readme.txt about the model and thats it!
	'''

	train_image, train_label, train = init_train_data()

	weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5 = gradient_descent(train_image, train_label, 5000, 0.1)

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

def forward_prop_5(image, weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, weights5, biases5, activation1, activation2, activation3, activation4):
	
	node0 = image

	network1 = weights1.dot(node0) + biases1
	node1 = activation1(network1)

	network2 = weights2.dot(node1) + biases2
	node2 = activation2(network2)

	network3 = weights3.dot(node2) + biases3
	node3 = activation3(network3)

	network4 = weights4.dot(node3) + biases4
	node4 = activation4(network4)

	network5 = weights5.dot(node4) + biases5
	node5 = softmax(network5) #THIS LAST NODE's ACTIVATION FUNCTION NEEDS TO BE SOFTMAX

	return network1, node1, network2, node2, network3, node3, network4, node4, network5, node5

def forward_prop_4(image, weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4, activation1, activation2, activation3):
	
	node0 = image

	network1 = weights1.dot(node0) + biases1
	node1 = activation1(network1)

	network2 = weights2.dot(node1) + biases2
	node2 = activation2(network2)

	network3 = weights3.dot(node2) + biases3
	node3 = activation3(network3)

	network4 = weights4.dot(node3) + biases4
	node4 = softmax(network4) #THIS LAST NODE's ACTIVATION FUNCTION NEEDS TO BE SOFTMAX

	return network1, node1, network2, node2, network3, node3, network4, node4

def test(test_image):

	# test_image, test_label, test = init_test_data()

	# tmp = test_image
	# print(test_image.shape)
	# print(test_image)
	# print(test_image.T[1].T.shape)
	# print(np.array([test_image.T[0]]))

	# test_image = np.array([test_image.T[2]]).T

	AW1 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/W1.npy")
	AB1 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/B1.npy")
	AW2 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/W2.npy")
	AB2 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/B2.npy")
	AW3 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/W3.npy")
	AB3 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/B3.npy")
	AW4 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/W4.npy")
	AB4 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/B4.npy")
	AW5 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/W5.npy")
	AB5 = np.load("EMNIST/EMNIST/100S-100R-100S-100S-62SM_76%/B5.npy")
	
	A_network1, A_node1, A_network2, A_node2, A_network3, A_node3, A_network4, A_node4, A_network5, A_node5 = forward_prop_5(test_image, AW1, AB1, AW2, AB2, AW3, AB3, AW4, AB4, AW5, AB5, sigmoid, ReLU, sigmoid, sigmoid)

	BW1 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/W1.npy")
	BB1 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/B1.npy")
	BW2 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/W2.npy")
	BB2 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/B2.npy")
	BW3 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/W3.npy")
	BB3 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/B3.npy")
	BW4 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/W4.npy")
	BB4 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/B4.npy")
	BW5 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/W5.npy")
	BB5 = np.load("EMNIST/EMNIST/120S-110R-100S-90R-62SM_77%/B5.npy")

	B_network1, B_node1, B_network2, B_node2, B_network3, B_node3, B_network4, B_node4, B_network5, B_node5 = forward_prop_5(test_image, BW1, BB1, BW2, BB2, BW3, BB3, BW4, BB4, BW5, BB5, sigmoid, ReLU, sigmoid, ReLU)

	CW1 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/W1.npy")
	CB1 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/B1.npy")
	CW2 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/W2.npy")
	CB2 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/B2.npy")
	CW3 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/W3.npy")
	CB3 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/B3.npy")
	CW4 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/W4.npy")
	CB4 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/B4.npy")
	CW5 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/W5.npy")
	CB5 = np.load("EMNIST/EMNIST/120S-130R-150S-120R-62SM_75%/B5.npy")

	C_network1, C_node1, C_network2, C_node2, C_network3, C_node3, C_network4, C_node4, C_network5, C_node5 = forward_prop_5(test_image, CW1, CB1, CW2, CB2, CW3, CB3, CW4, CB4, CW5, CB5, sigmoid, ReLU, sigmoid, ReLU)

	DW1 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/W1.npy")
	DB1 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/B1.npy")
	DW2 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/W2.npy")
	DB2 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/B2.npy")
	DW3 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/W3.npy")
	DB3 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/B3.npy")
	DW4 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/W4.npy")
	DB4 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/B4.npy")
	DW5 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/W5.npy")
	DB5 = np.load("EMNIST/EMNIST/200S-166S-133S-100S-62SM_80%/B5.npy")

	D_network1, D_node1, D_network2, D_node2, D_network3, D_node3, D_network4, D_node4, D_network5, D_node5 = forward_prop_5(test_image, DW1, DB1, DW2, DB2, DW3, DB3, DW4, DB4, DW5, DB5, sigmoid, sigmoid, sigmoid, sigmoid)

	EW1 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/W1.npy")
	EB1 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/B1.npy")
	EW2 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/W2.npy")
	EB2 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/B2.npy")
	EW3 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/W3.npy")
	EB3 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/B3.npy")
	EW4 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/W4.npy")
	EB4 = np.load("EMNIST/EMNIST/120R-100R-62R-62SM_80%/B4.npy")

	E_network1, E_node1, E_network2, E_node2, E_network3, E_node3, E_network4, E_node4 = forward_prop_4(test_image, EW1, EB1, EW2, EB2, EW3, EB3, EW4, EB4, ReLU, ReLU, ReLU)

	return parseLabel(np.argmax(A_node5 + B_node5 + C_node5 + D_node5 + E_node4))

	# print(A_node5)
	# print(B_node5)
	# print(C_node5)
	# print(D_node5)
	# print(E_node4)
	# print("result", A_node5 + B_node5 + C_node5 + D_node5 + E_node4)
	# print(np.argmax(A_node5 + B_node5 + C_node5 + D_node5 + E_node4))

	# plot_data(tmp.T, test_label, 2)
	
def parseLabel(label):
	alphabet = ["A","B","C","D","E",
				"F","G","H","I","J",
				"K","L","M","N","O",
				"P","Q","R","S","T",
				"U","V","W","X","Y",
				"Z",
				"a","b","c","d","e",
				"f","g","h","i","j",
				"k","l","m","n","o",
				"p","q","r","s","t",
				"u","v","w","x","y",
				"z"] 

	if label > 9:
		return alphabet[label-10]
	else :
		return label

def computerVision():

	cv2.namedWindow("A")
	cv2.namedWindow("B")
	vc = cv2.VideoCapture(1)

	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False

	while rval:

		h, w, c = frame.shape

		frame = frame[int(h/2-250):int(h/2+250), int(w/2-250):int(w/2+250)] #Image Size modify

		original = frame

		frame = cv2.GaussianBlur(frame,(15,15),4) #Gaussian Blur

		lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		l_channel, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl = clahe.apply(l_channel)
		limg = cv2.merge((cl,a,b))
		enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
		# frame = np.hstack((frame, enhanced_img)) #Increase Contrast
		frame = enhanced_img

		im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		thresh = 127
		frame = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1] #Convert to Black and White Image

		frame = cv2.bitwise_not(frame) #Invert Black and White

		# frame = cv2.flip(frame, 1)

		small_frame = cv2.resize(frame, (28,28)) #convert to 28*28

		r,c = small_frame.shape
		a = 0
		test_image = []

		for i in range(r):
			for j in range(c):
				test_image.append(small_frame[j,i]/255)
				# print(small_frame)

		test_image = np.array([test_image]).T

		# print()

		# plot_data(test_image.T,["A"],0)

		# cv2.imshow("asdf", small_frame)

		# cv2.putText()

		# frame = np.hstack((frame, test_image.reshape((28,28)).T))

		# label = 

		original = cv2.putText(original, "{}".format(test(test_image)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

		cv2.imshow("A", original)
		cv2.imshow("B", test_image.reshape((28,28)).T)

		# cv2.imshow("preview", cv2.flip(frame,1))
		# cv2.imshow("actual image", test_image.reshape((28,28)).T)

		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27:
			break

		# time.sleep(0.5)

	vc.release()
	cv2.destroyWindow("preview")

# train() #if you get rid of this, nothing will happen
# test()

# test(None)

computerVision()