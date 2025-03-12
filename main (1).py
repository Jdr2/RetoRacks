from ultralytics import YOLO
import multiprocessing
import cv2
import torch
torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    multiprocessing.freeze_support() # Ensure this is the first line in the if block

    # Load a model
    model = YOLO("Reto/best_70.pt")

    # Use the model
    model.train(data="Reto/config.yaml", epochs=100, imgsz=416 , optimizer="AdamW",patience=10, amp= True,lr0=0.001,weight_decay=0.0005)  # Train the model

    metrics = model.val()  # evaluate model performance on the validation set