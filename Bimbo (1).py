from ultralytics import YOLO
import numpy as np
import cv2
import time
import torch
torch.backends.cudnn.enabled = False

def numpy_mode(array):
    values, counts = np.unique(array, return_counts=True)
    index = np.argmax(counts)
    return values[index]

A_counter = 0
dectected_objs =[]
boxes_per_frame  = []
total_boxes = 0
boxes_needed = 16

model = YOLO("best_102.pt")

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    text = str(total_boxes )  # Text to overlay
    position = (30, 120)  # Text position (x, y) in pixels
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 4  # Font scale (size)
    color = (0, 255, 0)  # Text color in BGR (Green here)
    thickness = 6  # Thickness of the text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    results = model (source= frame, show= True, conf=0.65, save=False)


    for  result in results:
        # Extract the number of boxes detected in the current frame
        num_boxes_in_frame = len(result.boxes)
          # Store count per frame 
          

        if num_boxes_in_frame != 0:
                boxes_per_frame.append(num_boxes_in_frame)
                A_counter += 1
                if A_counter == 5:
                   total_boxes =  numpy_mode(boxes_per_frame)
                   A_counter = 0
                   boxes_per_frame = []  # Reset detections for the next batch
                else:
                    print("No objects detected in this frame.")
        elif num_boxes_in_frame == 0:
                total_boxes = 0

    

        

    print(f"Total number of boxes detected: {total_boxes}")
    print(f"Number of boxes per image: {boxes_per_frame}")