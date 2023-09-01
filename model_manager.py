from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Input, GlobalAveragePooling2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.models import Model

class ModelManager:
	def __init__(self):
		self.model = None

	def createCNN(self):
		base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
		x = base_model.output
		x = GlobalAveragePooling2D(input_shape=(1024, 1, 1))(x)

		x = Dense(2048, activation='relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.2)(x)

		x = Dense(512, activation='relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.2)(x)

		predictions = Dense(14, activation='sigmoid')(x)

		model = Model(inputs=base_model.input, outputs=predictions)

		for layer in base_model.layers:
			layer.trainable = False

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		self.model = model

	def getModel(self):
		if self.model != None:
			return self.model




