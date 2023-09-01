import os
import numpy as np
class Application:
	def __init__(self, dir):
		self.modelDriver = None
		self.mode = 0
		os.chdir(dir)

	def start(self):
		print("Please choose from one of the following numbers:\n")
		while(self.mode not in [1,2,3]):
			self.mode = input("1:[Re-train model]\n2:[Predict image]\n3:[Load custom Model]\n")
			try:
				self.mode = int(self.mode)
			except:
				print("Please enter a valid number")
				self.mode = -1

		if self.mode == 3:
			model_path = input("Please enter the full path for the model you wish to load, include the .h5 extension\n")
			image_path = input("Now enter the image you wish to diagnose\n")
			try:
				self._predict(image_path, model_path=model_path)
			except:
				print("Model provided does not exist")

		if self.mode == 1:
			save_filepath = input("Please enter the filepath you wish to save the trained model. Include .h5 as the file extension\n")
			self._initModelDriver(save_filepath)

		if self.mode == 2:
			image_path = input("Please enter the full filepath for the image you wish to diagnose.\n")
			self._predict(image_path)

	def _predict(self, image_path, model_path="CheXpert-v1.0-small/trained_model.h5"):
		from keras.models import load_model
		from keras.utils import load_img, img_to_array
		self.model = load_model(model_path, compile=True)

		img = load_img(image_path, target_size=(224, 224), color_mode="rgb")
		img = img_to_array(img)
		img = np.array([img])

		result = self.model.predict(img)

		self._interpretResults(result[0])

	def _interpretResults(self, results):
		assert(results.shape[0] == 14)
		results = list(results)
		print(self._decoder(results.index(max(results))))

	def _decoder(self, index):
		DECODER = {0:'No Finding', 1:'Enlarged Cardiomediastinum', 2:'Cardiomegaly', 3:'Lung Opacity', 
			4:'Lung Lesion', 5:'Edema', 6:'Consolidation', 7:'Pneumonia', 8:'Atelectasis', 9:'Pneumothorax', 10:'Pleural Effusion', 
			11:'Pleural Other', 12:'Fracture', 13:'Support Devices'}

		return DECODER[index]

	def _initModelDriver(self, save_location):
		from model_driver import ModelDriver
		self.modelDriver = ModelDriver()
		self.modelDriver.create_cnn()
		self.modelDriver.train_network(save_location)

def main():
	app = Application(".")
	app.start()

	

	

if __name__ == '__main__':
	main()