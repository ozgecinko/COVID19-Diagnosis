from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.metrics import AUC
import numpy as np

app = Flask(__name__)

diagnosis = {
	0: "Normal",
	1: "COVID-19",
}

model = load_model('cnn_model.h5', compile=False)

model.make_predict_function()

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(70,70))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1,70,70,3)

	predict_x = model.predict(test_image) 
	classes_x = np.argmax(predict_x,axis=1)

	return diagnosis[classes_x[0]]


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == "POST":
		img = request.files['my_image']
		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template(
		"index.html", prediction = predict_result, img_path = img_path
		)


if __name__ == "__main__":
    app.run(debug=True)