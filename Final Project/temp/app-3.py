from your_image_generation_module import create_video_from_images

# ...

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # ... (Keep the previous code for processing the audio file and extracting keywords)

        images = generate_images(keywords)

        # Create a video from the generated images
        unique_video_filename = f"{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_video_filename)
        create_video_from_images(images, video_path)

        response = {
            'keywords': keywords,
            'video_url': f"/{app.config['UPLOAD_FOLDER']}/{unique_video_filename}"
        }

        return jsonify(response)

# ...
