import os
import cv2
import base64
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import string
import random

# Load the trained model
model = tf.keras.models.load_model(os.path.join('models', 'updated_imageclassifier2.h5'))

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Function to generate a random filename
def generate_random_filename():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        if file:
            # Read the image and preprocess it
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            resize = tf.image.resize(img, (256, 256))
            processed_image = np.expand_dims(resize / 255, 0)

            # Make a prediction using the model
            prediction = model.predict(processed_image)
            predicted_class = "Miku" if prediction[0][0] < 0.5 else "Not Miku"

            # Encode the image as a base64 string
            _, encoded_image = cv2.imencode('.jpg', img)
            base64_image = base64.b64encode(encoded_image).decode('utf-8')

            # Render the result page with the prediction and the base64 encoded image
            return render_template('result.html', predicted_class=predicted_class, base64_image=base64_image)

    # Render the home page for uploading an image
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
