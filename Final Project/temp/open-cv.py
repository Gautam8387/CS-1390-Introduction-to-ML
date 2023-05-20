import cv2
import os

# Change these variables to match your input and output directories and video parameters
IMAGE_DIR = "./images/" # directory containing input images
VIDEO_FILE = "output.avi" # name of output video file
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
