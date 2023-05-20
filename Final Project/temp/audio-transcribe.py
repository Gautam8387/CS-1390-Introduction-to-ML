# from huggingsound import SpeechRecognitionModel

# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
# audio_paths = ["./data/chunk_1.wav"]#, "./data/chunk_0.wav"]

# transcriptions = model.transcribe(audio_paths)
# print(transcriptions)

import requests

API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("./data/chunk_1.wav")
# Write the output to a file
print(output['text'])