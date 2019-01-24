# I fucked up, but saved myself some time. Cannot change batch_size at runtime,
# Or I don't know enough keras to do this
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback

class BatchSizeFinder(Callback):
	'''
	A simple callback for finding the optimal learning rate range for your model + dataset

	Usage:
	bfinder = BatchSizeFinder(min_size=2, max_size=64, steps_per_epoch=np.ceil(epoch_size/batch), epochs=3)
	model.fit(x_train, y_train, callbacks=[bfinder])

	bfinder.plot_loss()
	'''

	def __init__(self, min_size=2, max_size=64, steps_per_epoch=None, epochs=None):
		super().__init__()
		self.min_size = min_size
		self.max_size = max_size
		self.total_iterations = steps_per_epoch * epochs
		self.iteration = 0
		self.history = {}

	def cbs(self):
		'''Calculate the batch size'''
		x = # fuck, idk