from huggingsound import SpeechRecognitionModel
import moviepy.editor as mp
from tqdm import tqdm
import enchant as en
import numpy as np
import requests
import spacy
import cv2
import os

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {}
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

def get_data(audio_file, topic):
    # Audio Transcription
    model_audio = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    audio_paths = [audio_file]
    transcription = model_audio.transcribe(audio_paths)

    # Spell check and replace incorrect words
    dict = en.Dict("en_US")
    sentence = transcription[0]['transcription']
    for words in sentence.split():
        if not dict.check(words):
            # If not in dictionary, replace with a similar word
            suggestions = dict.suggest(words)
            if len(suggestions) > 0:
                sentence = sentence.replace(words, suggestions[0])
            else:
                sentence = sentence.replace(words, "word")
    print("Spell Check Complete.")

    # NLP
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    nouns = [token.text for token in doc if token.ent_type_ == "GPE" or token.ent_type_ == "LOC" or token.pos_ == "NOUN"]
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    print("NLP Complete.")

    # Image Generation
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": "Bearer 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"}

    for i in range(max(len(nouns), len(adjectives), len(verbs))):
        # make a prompt using the nouns, adjectives, and verbs
        prompt = topic
        prompt += nouns[i % len(nouns)]
        prompt += " " + adjectives[i % len(adjectives)]
        prompt += " " + verbs[i % len(verbs)]
        # Query the model
        image_bytes = query({"inputs": prompt})
        # Save the image
        with open(f"./images/image_{i}.png", "wb") as f:
            f.write(image_bytes)
        print(f"Image {i} saved.")

    # Video Generation
    IMAGE_DIR = "./images/" # directory containing input images
    VIDEO_FILE = "./output/output.avi" # name of output video file
    FRAME_RATE = 25 # number of frames per second in output video
    FRAME_SIZE = (640, 480) # size of output video frames
    IMAGE_DISPLAY_TIME = 2 # time to display each image in seconds

    # Get a list of all the image files in the input directory
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]

    # Sort the image files by filename
    image_files.sort()

    # Calculate the number of frames to display each image based on the FRAME_RATE and IMAGE_DISPLAY_TIME
    frames_per_image = int(FRAME_RATE * IMAGE_DISPLAY_TIME)

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_FILE, fourcc, FRAME_RATE, FRAME_SIZE)

    # Loop through the image files and write each image to the output video multiple times
    for image_file in tqdm(image_files, desc="Generating video"):
        image = cv2.imread(image_file)
        # Resize the image to match the output frame size
        image = cv2.resize(image, FRAME_SIZE)
        for i in range(frames_per_image):
            out.write(image)

    # Release the VideoWriter object
    out.release()

    # Clean up the images directory
    # for f in image_files:
    #     os.remove(f)
    print("Video saved as", VIDEO_FILE)
    # Return the output video file path
    
    import moviepy.editor as mp

    # Video file in .avi format
    video_file = VIDEO_FILE

    # Audio file in .wav format to be added as background sound
    audio_file = audio_file

    # Load the video file as a VideoFileClip object
    video_clip = mp.VideoFileClip(video_file)
    # Load the audio file as an AudioFileClip object
    audio_clip = mp.AudioFileClip(audio_file)

    # durations
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration

    # If the video is longer than the audio, cut the video
    if audio_duration < video_duration:
        video_clip = video_clip.subclip(0, audio_duration)
    # If the audio is longer than the video, loop the video
    elif audio_duration > video_duration:
        video_clip = mp.concatenate_videoclips([video_clip] * int(np.ceil(audio_duration / video_duration)))
        video_clip = video_clip.subclip(0, audio_duration)

    final_clip = video_clip.set_audio(audio_clip)#.set_duration(video_clip.duration))

    # Set the audio of the video clip as the audio clip with the audio from the beginning of the video
    # final_clip = video_clip.set_audio(audio_clip.set_duration(video_clip.duration))

    # Save the final clip with the added background sound as a .avi file
    # final_clip.write_videofile("./output/final_video.avi", codec="prores")#"mpeg4")   
    # Save the final clip with the added background sound as a .mp4 file
    final_clip.write_videofile("./output/final-video.mp4", codec="libx264")#"mpeg4")
        
    return final_clip


