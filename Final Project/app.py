from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from get_data import get_data
# from pkgutil import get_data
import os

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Secret key for flash messages
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Allowed file types for upload
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Directory to save uploaded files
UPLOAD_FOLDER = './data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Get the topic from the form data
    topic = request.form['topic']

    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))

    file = request.files['file']

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash('File uploaded successfully')

        # Run the get_data function with the uploaded file and topic
        get_data(filepath, topic)

        # Return the generated video file
        # send_file('./output/final-output.avi', as_attachment=True)
        # return send_file('./output/output.avi', as_attachment=True)
        
        return redirect(url_for('video'))
        # return redirect(url_for('video', filename='final-video.mp4'))
    else:
        flash('Invalid file type')
        return redirect(url_for('home'))

# Route for video page
@app.route('/video')
def video():
    return render_template('video.html')
    # video_url = url_for('static', filename='output/final-video.mp4')
    # return render_template('video.html', video_url=video_url)

# Function to check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)