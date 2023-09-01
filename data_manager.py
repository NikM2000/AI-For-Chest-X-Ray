import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
class DataManager:
	def __init__(self, file_path):
		if os.path.isfile(file_path):
			self.path = file_path
			self.data = None
		else:
			self.path = None
			raise Exception("File path provided does not exist.")

	def validatePath(self):
		return os.path.isfile(self.path) and self.path[-3:] == "csv"

	def csv_to_dataframe(self):
		if (self.validatePath()):
			self.data = pd.read_csv(self.path, low_memory=False).fillna(0)
			return self.data

	def encodeTargetLabels(self, targetLabels):
		labels = self.data['labels'].values
		encoder = MultiLabelBinarizer(classes=targetLabels)	
		targetLabels = encoder.fit_transform(labels)
		self.data['targetLabels'] = targetLabels.tolist()


	def filterValues(self, column_name, value):
		if column_name in self.data.columns:
			self.data.drop(self.data[self.data[column_name] == value].index, inplace = True)
		else:
			print("Column_name provided does not exist.")

	def encodeUncertainty(self, uncertainty_matrix):
		if self._validateLabels(uncertainty_matrix.keys()):
			for feature in uncertainty_matrix.keys():
				condition = (self.data[feature] == -1)
				if uncertainty_matrix[feature] == True:
					self.data[feature] = self.data[feature].mask(condition, 1)
				else:
				 	self.data[feature] = self.data[feature].mask(condition, 0)
		else:
			print("Invalid labels in uncertainty matrix provided.")


	def _obtainLabelValue(self, row, labels):
		result = []
		for feature in labels:
			if row[feature] in [-1, 1]:
				result.append(feature)

		if result == []:
			result.append("No Finding")
		return ','.join(result)

	def resetIndex(self):
		self.data.reset_index(inplace=True, drop=True)

	def dropColumns(self, columns):
		if self._validateLabels(columns):
			for column in columns:
				self.data.drop(column, axis=1, inplace=True)

	def getData(self):
		return self.data

	def getColumnNames(self):
		return [col for col in self.data.columns]

	def getDataShape(self):
		return self.data.shape

	def getSubsetByColumns(self, columns):
		if self._validateLabels(columns):
			return self.data[columns]

	def _validateLabels(self, labels):
		for label in labels:
			if label not in self.data: return False
		return True

