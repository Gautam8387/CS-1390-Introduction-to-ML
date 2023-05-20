# pip install opencv-python opencv-python-headless
# Next, create a new Python function in your your_image_generation_module to generate a video from the given images:
import cv2

def create_video_from_images(image_list, output_path, fps=30):
    height, width, layers = image_list[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for i in range(len(image_list)):
        out.write(image_list[i])
    
    out.release()
