# Python Script to Convert an audio file into chunks of 30 seconds each
import os
import librosa
import soundfile as sf

def audio_chunk(audio_file, chunk_length=30):
    # Load the audio file
    y, sr = librosa.load(audio_file)
    # Get the length of the audio file
    audio_length = librosa.get_duration(y=y, sr=sr)
    # Calculate the number of chunks
    n_chunks = int(audio_length // chunk_length)
    # Calculate the starting and ending sample for each chunk
    # Save each chunk as a wav file
    for i in range(n_chunks):
        start_sample = [int(chunk_length * sr * i) for i in range(n_chunks)]
        end_sample = [int(chunk_length * sr * (i + 1)) for i in range(n_chunks)]
        y_chunk = y[start_sample[i]:end_sample[i]]
        sf.write('chunk_{}.wav'.format(i), y_chunk, sr)

        
    
if __name__ == '__main__':
    audio_chunk('../data/eg-2/video-2.wav', 30)