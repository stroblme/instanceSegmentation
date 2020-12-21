import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from utils import *
import time

# Set this variable to False if you want to view a background image
# and also set the path to your background image
# Background Image Source: https://pixabay.com/photos/monoliths-clouds-storm-ruins-sky-5793364/
BLUR = True
BG_PTH = "bg1.jpg"

# Load the DeepLabv3 model to memory
model = utils.load_model()

# Read the background image to memory
bg_image = cv2.imread(BG_PTH)
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

# Start a video cam session
video_session = cv2.VideoCapture(0)

# Define a blurring value kernel size for cv2's Gaussian Blur
blur_value = (51, 51)

# Define two axes for showing the mask and the true video in realtime
# And set the ticks to none for both the axes
fig, ax1 = plt.subplots(1, 1, figsize = (15, 8))

ax1.set_xticks([])
ax1.set_yticks([])


# Create two image objects to picture on top of the axes defined above
im1 = ax1.imshow(utils.grab_frame(video_session))

# Switch on the interactive mode in matplotlib
plt.ion()
plt.show()

# Read frames from the video, make realtime predictions and display the same
while True:
    frame = utils.grab_frame(video_session)

    # Ensure there's something in the image (not completely blacnk)
    if np.any(frame):
        start = time.time()

        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width, height, channels = frame.shape
        labels = utils.get_pred(frame, model)
        
        predT = time.time()
        print(f"predT = {predT - start}")


        # Wherever there's empty space/no person, the label is zero 
        # Hence identify such areas and create a mask (replicate it across RGB channels)
        mask = labels == 0
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)

        maskT = time.time()
        print(f"maskT = {maskT - predT}")



        # Apply the Gaussian blur for background with the kernel size specified in constants above
        blur = cv2.GaussianBlur(frame, blur_value, 0)
        frame[mask] = blur[mask]
        ax1.set_title("Blurred Video")

        blurT = time.time()
        print(f"blurT = {blurT - maskT}")

        
        # Set the data of the two images to frame and mask values respectively
        im1.set_data(frame)
        # im2.set_data(mask * 255)

        dispT = time.time()
        print(f"dispT = {dispT - blurT}")

        plt.pause(0.01)
        
    else:
        break

# Empty the cache and switch off the interactive mode
torch.cuda.empty_cache()
plt.ioff()
