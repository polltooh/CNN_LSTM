import numpy as np
import cv2


class DATA_QUEUE():
	def __init__(self):
		""" data shape:
			[20, 10000, 64, 64]: 
		"""
		self.data = np.load("mnist_test_seq.npy")

		print("after loading the data")
		# self.data = np.load("temp.npy")
		train_data_num = int(self.data.shape[1] * 0.7)
		self.train_data = self.data[:,0:train_data_num,:,:]
		self.test_data = self.data[:,train_data_num:,:,:]
		self.data = self.train_data
		self.data_shape = self.data.shape

		self.train_start_index = 0
		self.test_start_index = 0

		self.train_data_num = train_data_num
		self.test_data_num = self.test_data.shape[1]

	def get_next_batch_train(self, batch_size, reshape = True, expand_dim = -1):
		""" 
		return: 
			if reshape == false and expand_dim = 4
				[20,batch_size, 64, 64, 1]
		"""
		end_index = self.train_start_index + batch_size
		batch_shape = list(self.data.shape)
		batch_shape[1] = batch_size
		batch_data = np.zeros((batch_shape), np.float32)
		if end_index < self.train_data_num:
			batch_data = self.data[:,self.train_start_index:end_index,:,:]
		else:
			batch_data[:,0:(self.train_data_num - self.train_start_index),:,:] = \
				self.data[:,self.train_start_index:self.train_data_num,:,:]
			end_index = end_index - self.train_data_num
			batch_data[:,(self.train_data_num - self.train_start_index):,:,:] = \
				self.data[:,0:end_index,:,:]
		if reshape:
			batch_data = np.reshape(batch_data,(20,batch_size,-1))
		if expand_dim != -1:
			batch_data = np.expand_dims(batch_data, expand_dim)
		self.train_start_index = end_index
		return batch_data

	def get_next_batch_test(self, batch_size, reshape =True, expand_dim = -1):
		end_index = self.test_start_index + batch_size
		batch_shape = list(self.test_data.shape)
		batch_shape[1] = batch_size
		batch_data = np.zeros((batch_shape), np.float32)
		if end_index < self.test_data_num:
			batch_data = self.test_data[:,self.test_start_index:end_index,:,:]
		else:
			batch_data[:,0:(self.test_data_num - self.test_start_index),:,:] = \
				self.test_data[:,self.test_start_index:self.test_data_num,:,:]
			end_index = end_index - self.test_data_num
			batch_data[:,(self.test_data_num - self.test_start_index):,:,:] = \
				self.test_data[:,0:end_index,:,:]

		if reshape:
			batch_data = np.reshape(batch_data,(20,batch_size,-1))
			
		if expand_dim != -1:
			batch_data = np.expand_dims(batch_data, expand_dim)

		self.test_start_index = end_index
		return batch_data

	def display_digit(self,img):
		cv2.imshow("test", img)
		cv2.waitKey(0)

if __name__ == "__main__":
	data_queue = DATA_QUEUE()
	while (1):
		data = data_queue.get_next_batch_test(10)
		print(data.shape)
		# print(data.shape)
