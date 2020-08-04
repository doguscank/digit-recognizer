import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, MaxPooling2D, Activation, Softmax, Flatten, Dense, BatchNormalization, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

class Recognizer:
	def __init__(self, train_path = './dataset/train.csv'):
		self.train_path = train_path
		self.i = 1
		self.nrows = 0
		self.classes = range(10)

	#nrows: Import "nrows" rows. Use -1 to import all
	def import_dataset(self, nrows = 50):
		if nrows == -1:
			self.train_set = pd.read_csv(self.train_path)
			self.nrows = len(self.train_set)
		else:
			self.train_set = pd.read_csv(self.train_path, nrows = nrows)
			self.nrows = nrows

	def import_testset(self, nrows = 50):
		if nrows == -1:
			self.test_set = pd.read_csv('./dataset/test.csv')
			self.reshape_testset(nrows)
		else:
			self.test_set = pd.read_csv('./dataset/test.csv', nrows = nrows)
			self.reshape_testset(nrows)

	def reshape_dataset(self):
		self.train_y = self.train_set['label'].values
		self.train_y = np.asarray([np.equal(y, self.classes).astype(int) for y in self.train_y])
		self.train_X = [(self.train_set.values[:, 1:].reshape(self.nrows, 28, 28) / 255.0)]
		self.train_X = np.stack((self.train_X, ) * 3, axis = -1)
		self.test_X = self.train_X[0][:-1000]
		self.train_X = self.train_X[0][:41000]
		self.test_y = self.train_y[:-1000]
		self.train_y = self.train_y[:41000]

	def reshape_testset(self, nrows):
		self.testset_X = [(self.test_set.values.reshape(nrows, 28, 28) / 255.0)]
		self.testset_X = np.stack((self.testset_X, ) * 3, axis = -1)

	#i: Index of the data to get from dataset to visualize
	#_print: Print data as array
	def visualize_ith_input(self, i = 0, _print = False):
		img = np.array([recognizer.train_set.values[i][1:]])
		img = img.reshape((28, 28, 1)) / 255.0

		if _print: print(img)

		cv2.imshow("img", img)
		cv2.waitKey(0)

	def init_model(self):
		self.model = Sequential()
		self.model.add(self.CNN_input())

		return self.model

	def CNN_input(self):
		cnn_input = Conv2D(4, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu', input_shape = (28, 28, 3), name = 'cnn_input_layer')
		return cnn_input

	def softmax_output(self):
		self.model.add(Softmax())

		return self.model

	def add_residual_block(self, num_filter, f, s, paddings, pool_size):
		[f1, f2, f3] = f
		[s1, s2, s3] = s
		[p1, p2, p3] = paddings

		cnn_layer_1 = Conv2D(num_filter, (f1, f1), strides = (s1, s1), padding = p1, name = 'cnn_block_{}_1'.format(self.i))
		batch_normalization_1 = BatchNormalization(name = 'bn_block_{}_1'.format(self.i))
		relu_1 = Activation('relu', name = 'relu_block_{}_1'.format(self.i))

		cnn_layer_2 = Conv2D(num_filter, (f2, f2), strides = (s2, s2), activation = 'relu', padding = p2, name = 'cnn_block_{}_2'.format(self.i))				
		batch_normalization_2 = BatchNormalization(name = 'bn_block_{}_2'.format(self.i))
		relu_2 = Activation('relu', name = 'relu_block_{}_2'.format(self.i))

		cnn_layer_3 = Conv2D(num_filter, (f3, f3), strides = (s3, s3), padding = p3, name = 'cnn_block_{}_3'.format(self.i))		
		batch_normalization_3 = BatchNormalization(name = 'bn_block_{}_3'.format(self.i))

		#add = Add()([batch_normalization_1, batch_normalization_3])
		relu_3 = Activation('relu', name = 'relu_block_{}_3'.format(self.i))

		max_pool = MaxPooling2D((pool_size, pool_size))

		self.model.add(cnn_layer_1)
		self.model.add(batch_normalization_1)
		self.model.add(relu_1)
		self.model.add(cnn_layer_2)
		self.model.add(batch_normalization_2)
		self.model.add(relu_2)
		self.model.add(cnn_layer_3)
		self.model.add(batch_normalization_3)
		#self.model.add(add)
		self.model.add(relu_3)
		self.model.add(max_pool)

		self.i += 1

		return self.model

	def flatten_layer(self):
		self.model.add(Flatten())

		return self.model

	def reshape_layer(self):
		self.model.add(Reshape((90, )))

	def add_dense(self):
		dense_1 = Dense(90, activation = 'relu')
		dense_2 = Dense(64, activation = 'relu')
		dense_3 = Dense(32, activation = 'relu')
		dense_output = Dense(10, activation = 'softmax')

		self.model.add(dense_1)
		self.model.add(dense_2)
		self.model.add(dense_3)
		self.model.add(dense_output)

		return self.model

	def resize_img(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		resized = cv2.resize(gray, (28, 28))
		cv2.imshow('resize_img', resized)
		cv2.waitKey(0)
		resized = np.stack(([resized], ) * 3, axis = -1)

		return resized

if __name__ == '__main__':
	recognizer = Recognizer()
	recognizer.import_dataset(-1)

	"""
	recognizer.reshape_dataset()
	recognizer.init_model()
	recognizer.add_residual_block(10, (2, 2, 1), (2, 2, 1), ('valid', 'valid', 'same'), 2)
	recognizer.add_residual_block(10, (2, 2, 1), (1, 1, 1), ('same', 'same', 'same'), 1)
	#recognizer.flatten_layer()
	recognizer.reshape_layer()
	recognizer.add_dense()
	#recognizer.softmax_output()

	print(recognizer.train_y.shape)

	recognizer.model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	recognizer.model.fit(recognizer.train_X, recognizer.train_y, epochs = 50, batch_size = 256)

	results = recognizer.model.evaluate(recognizer.test_X, recognizer.test_y, batch_size = 256)
	print(results)

	recognizer.model.save('./saved_models/model_3')
	"""

	
	recognizer.init_model()
	recognizer.model = tf.keras.models.load_model('./saved_models/model_2')
	

	
	recognizer.import_testset()
	np.random.shuffle(recognizer.testset_X[0])

	for i, img in enumerate(recognizer.testset_X[0]):
		#print(img)
		pred = recognizer.model.predict(np.asarray([img]))
		print("{}th image: {}".format(i, np.argmax(pred)))
		cv2.imshow("test_img", img)
		cv2.waitKey(0)
	
	

	"""
	test_img = cv2.imread('./dataset/test_png/3_1.png')
	test_img = recognizer.resize_img(test_img)

	pred = recognizer.model.predict(test_img)
	print(pred)
	"""
	
	#recognizer.visualize_ith_input(1)