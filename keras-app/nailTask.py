import numpy as np
import io
import os
os.environ['KERAS_BACKEND'] = 'theano'
import tensorflow as tf
import keras
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import flask
from keras.models import model_from_json


app = flask.Flask(__name__)
model = None


def load_model():
	# load the pre-trained Keras model (here we are using a model
	# We  load our own model that is an mobile net trained on image net 
	# Then transfer learning was used to train it on the two nail classes
	global model
	
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# to access the graph 
			global graph
			graph = tf.get_default_graph()

			#get predictions for single image
			with graph.as_default():
				preds = model.predict(image)
			
			data["predictions"] = []

			
			#prepare predictions for both classes as readable output
			bad = {"label": "bad", "probability": float(preds[0][0])}
			data["predictions"].append(bad)
			good = {"label": "good", "probability": float(preds[0][1])}
			data["predictions"].append(good)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)
	
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0')