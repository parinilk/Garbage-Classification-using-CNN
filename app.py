from flask import Flask, render_template, request, redirect, url_for, send_file, make_response, session
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO 




pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = Flask(__name__)
app.secret_key = 'lalabonjour'



# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = os.path.join(app.root_path, 'static')
app.config['STATIC_FOLDER_UPLOADS'] = os.path.join(app.root_path,'uploads')

from flask import send_from_directory



# Load your DenseNet model
model = load_model('models\densenet_model.h5')

# Mapping of prediction integers to class names
class_names = {
    0: 'Cardboard',
    1: 'Glass',
    2: 'Metal',
    3: 'Paper',
    4: 'Plastic',
    5: 'Trash'
}

def predict_class(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Preprocess the image
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    return predicted_class  # Assuming the classes are represented as integers

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class = predict_class(file_path)
            return redirect(url_for('result', prediction=predicted_class))
    return render_template('upload.html')

@app.route('/result/<int:prediction>')
def result(prediction):
    class_name = class_names.get(prediction, 'Unknown')
    template_name = f'result_{class_name.lower()}.html'
    return render_template(template_name)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_ai_image(prompt):
    output = pipe(prompt, num_infernece_steps=50)
    print(output.keys())
    image = output["images"][0]
    return image

def save_image(image):

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])


    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_image.png')
    image.save(image_path)

    return image_path
@app.route('/generate_image', methods=['GET', 'POST'])
def generate_image():
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_image = generate_ai_image(prompt)

        image_path = save_image(generated_image)

        predicted_class = predict_class(image_path)
        
        predicted_class_name = class_names[predicted_class]
        
        session['predicted_class_name'] = predicted_class_name
        
        return redirect(url_for('dresult'))



    return render_template('generate_image.html')

@app.route('/result')
def dresult():
    predicted_class_name = session.get('predicted_class_name')

    if predicted_class_name:
        return render_template('result.html', predicted_class_name=predicted_class_name)
    else:
        return redirect(url_for('generate_image'))


@app.route('/generated_image.png')
def serve_generated_image():
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_image.png')
    return send_file(image_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
