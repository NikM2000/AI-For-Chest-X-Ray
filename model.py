from data_manager import DataManager
import os
from keras_preprocessing.image import ImageDataGenerator
from model_manager import ModelManager
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
class ChexPertModel:
	def __init__(self, dataManager):
		self.UNCERTAINTY = {'Enlarged Cardiomediastinum':True, 'Cardiomegaly':True, 'Lung Opacity':False, 
		'Lung Lesion':False, 'Edema':True, 'Consolidation':False, 'Pneumonia':True, 'Atelectasis':True, 'Pneumothorax':True, 'Pleural Effusion':True, 
		'Pleural Other':False, 'Fracture':False, 'Support Devices':False}
		self.dataManager = dataManager
		self.dataManager.csv_to_dataframe()
		self.dataManager.filterValues("Frontal/Lateral", "Lateral")
		self.dataManager.dropColumns(["Sex", "Age", "Frontal/Lateral", "AP/PA"])
		self.dataManager.resetIndex()

		self.dataManager.encodeUncertainty(self.UNCERTAINTY)
		self.data = dataManager.getData()

		self._train_valid_split(0.2)

		self.modelManager = ModelManager()

	def createImageGenerators(self, batch_size, target_size):
		train_datagen = ImageDataGenerator(rescale=1./255)
		valid_datagen = ImageDataGenerator(rescale=1./255)

		train_generator = train_datagen.flow_from_dataframe(
			dataframe=self.train_data,
			x_col="Path",
			y_col=list(self.train_data.columns[1:15]),
			batch_size=batch_size,
			seed=42,
			shuffle=True,
			target_size=target_size,
			class_mode="raw",
			)

		valid_generator = valid_datagen.flow_from_dataframe(
			dataframe=self.valid_data,
			x_col="Path",
			y_col=list(self.valid_data.columns[1:15]),
			batch_size=batch_size,
			seed=42,
			shuffle=True,
			target_size=target_size,
			class_mode="raw",
			)

		return (train_generator, valid_generator)

	def getTargetLabels(self):
		return ["No Finding"] + list(self.UNCERTAINTY.keys())

	def getSubsetByColumns(self, columns):
		return self.dataManager.getSubsetByColumns(columns)

	def getData(self):
		return self.data

	def getModelManager(self):
		return self.modelManager

	def _train_valid_split(self, valid_split):
		data = self.data
		valid_length = round(data.shape[0] * valid_split)
		self.train_data = data[0:data.shape[0] - valid_length]
		self.valid_data = data[self.train_data.shape[0]:]