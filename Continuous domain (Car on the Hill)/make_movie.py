import cv2
import os
import re

# XXX: the images must have been generated already, and be in the current directory

def convert(str):
    return int("".join(re.findall(r"\d*", str)))


image_folder = '.'
video_name = 'video2.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

images.sort(key=convert)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
