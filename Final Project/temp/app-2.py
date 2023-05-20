from flask import Flask, render_template, request, send_file, jsonify
import os
import io
import uuid
from your_speech_recognition_module import process_audio_file
from your_ner_module import extract_keywords
from your_image_generation_module import generate_images

app = Flask(__name__)

# Set upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # Get the uploaded audio file
        audio_file = request.files['audio_file']
        if audio_file:
            transcript = process_audio_file(audio_file)
            keywords = extract_keywords(transcript)
            image = generate_images(keywords)

            # Save image to the UPLOAD_FOLDER with a unique filename
            unique_filename = f"{uuid.uuid4().hex}.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            image.save(image_path)

            response = {
                'keywords': keywords,
                'image_url': f"/{app.config['UPLOAD_FOLDER']}/{unique_filename}"
            }

            return jsonify(response)

    return 'Error processing the request', 400

if __name__ == '__main__':
    app.run(debug=True)
