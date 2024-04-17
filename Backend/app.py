from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
from keras.models import load_model, save_model

# Load the existing model
#model = load_model(r'C:\Users\Y PAVANI\LUNG CANCER DETECTION PROJECT.ipynb')

# Save the model in HDF5 format
#save_model(model, 'C:/Users/Y PAVANI/Documents.h5')
model = load_model(r'C:\Users\Y PAVANI\Documents\modelfb1.h5')

# Define the classes
#CLASSES = {0: 'normal', 1: 'pneumonia'}
CLASSES = {0 : 'benign', 1:'malignant', 2:'normal'}

# Define the image size
IMAGE_SIZE = (150, 150)


def preprocess_image(image):
    # Preprocess the image as required by the model
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']

        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = preprocess_image(image)

        # Make the prediction
        prediction = model.predict(processed_image)
        predicted_class = CLASSES[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)