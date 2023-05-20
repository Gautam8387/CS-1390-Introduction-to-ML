# Import Libraries
from huggingsound import SpeechRecognitionModel
from tqdm import tqdm
import numpy as np
import enchant as en
import requests
import spacy
import cv2
import os

# Ask User for Input
print("Please enter the TOPIC of your script:")
topic = input()

### Audio Transcription ###
model_audio = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["./data/chunk_7.wav"]#, "./data/chunk_0.wav"]
transcription = model_audio.transcribe(audio_paths)
# print(transcription[0]['transcription'])

### NLP ###
dict = en.Dict("en_US")
sentence = transcription[0]['transcription']
for words in sentence.split():
    if not dict.check(words):
        # If not in dictory, replace with a similar word
        suggestions = dict.suggest(words)
        if len(suggestions) > 0:
            sentence = sentence.replace(words, suggestions[0])
        else:
            sentence = sentence.replace(words, "word")
# print(sentence)

# Load the spacy model
nlp = spacy.load("en_core_web_sm")
# sentence = "The quick brown fox jumped over the lazy dog"#
# sentence = "Paris is the capital of France and it is a beautiful city."
sentence = sentence #transcription[0]['transcription']
doc = nlp(sentence)
nouns = []
adjectives = []
verbs = []
for token in doc:
    if token.ent_type_ == "GPE" or token.ent_type_ == "LOC" or token.pos_ == "NOUN":#if token.dep_ == "nsubj"or token.dep_ == "dobj" or token.pos_ == "NOUN": #  or token.dep_ == "pobj" 
        nouns.append(token.text)
    elif token.pos_ == "ADJ":
        adjectives.append(token.text)
    elif token.pos_ == "VERB":
        verbs.append(token.text)
# print("Nouns: ", nouns)
# print("Adjectives: ", adjectives)
# print("Verbs: ", verbs)

### Image Generation ###
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

for i in range(max(len(nouns), len(adjectives), len(verbs))):
	# make a prompt using the nouns, adjectives, and verbs
	prompt = topic
	prompt += nouns[i%len(nouns)]
	prompt += " " + adjectives[i%len(adjectives)]
	prompt += " " + verbs[i%len(verbs)]
    # Query the model
	image_bytes = query({"inputs": prompt})
	# Save the image
	with open(f"./images/image_{i}.png", "wb") as f:
		f.write(image_bytes)

### Video Generation ###
# Change these variables to match your input and output directories and video parameters
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

# Loop through the image files and add them to the output video
for image_file in image_files:
    # Load the image file
    img = cv2.imread(image_file)
    
    # Resize the image to match the output frame size
    img = cv2.resize(img, FRAME_SIZE)
    
    # Add each frame of the image to the output video frames_per_image times
    for i in range(frames_per_image):
        out.write(img)

# Release the VideoWriter object and close the output file
out.release()
print("Video saved as", VIDEO_FILE)