import os
from data_manager import DataManager
from model import ChexPertModel
class ModelDriver():
	def __init__(self):
		data = DataManager("CheXpert-v1.0-small/train.csv")
		self.model = ChexPertModel(data)
		self.trainGenerator, self.validGenerator = self.model.createImageGenerators(32, (224, 224))
		self.modelManager = self.model.getModelManager()
		self.neuralNet = None

	def create_cnn(self):
		self.modelManager.createCNN()
		self.neuralNet = self.modelManager.getModel()
		return self.neuralNet

	def train_network(self, filepath):
		model = self.neuralNet

		train_step_size = self.trainGenerator.n//self.trainGenerator.batch_size
		valid_step_size = self.validGenerator.n//self.validGenerator.batch_size

		history = model.fit(self.trainGenerator, steps_per_epoch=train_step_size, validation_data=self.validGenerator, validation_steps=valid_step_size, epochs=3)

		model.save(filepath)

	def modelExists(self):
		return type(self.neuralNet) != None