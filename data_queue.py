import numpy as np
import cv2


class DATA_QUEUE():
	def __init__(self):
		# self.data = np.load("mnist_test_seq.npy")
		self.data = np.load("temp.npy")
		print("after loading the data")
		self.data_shape = self.data.shape
		self.start_index = 0
		self.data_num = self.data.shape[1]

	def get_next_batch(self, batch_size):
		end_index = self.start_index + batch_size
		batch_shape = list(self.data.shape)
		batch_shape[1] = batch_size
		batch_data = np.zeros((batch_shape), np.float32)
		if end_index < self.data_num:
			batch_data = self.data[:,self.start_index:end_index,:,:]
		else:
			batch_data[:,0:(self.data_num - self.start_index),:,:] = \
				self.data[:,self.start_index:self.data_num,:,:]
			end_index = end_index - self.data_num
			batch_data[:,(self.data_num - self.start_index):,:,:] = \
				self.data[:,0:end_index,:,:]
		batch_data = np.reshape(batch_data,(20,batch_size,-1))
		self.start_index = end_index
		return batch_data

	def display_digit(self,img):
		cv2.imshow("test", img)
		cv2.waitKey(0)

if __name__ == "__main__":
	data_queue = DATA_QUEUE()
	data = data_queue.get_next_batch(10)
	print(data.shape)
