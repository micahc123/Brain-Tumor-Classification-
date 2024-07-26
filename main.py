from ultralytics import YOLO
import numpy as np
import os
import cv2  

model = YOLO("runs/classify/train/weights/best.pt")

folder_input_path = "test_imgs"
folder_output_path = "annotated_imgs"  

if not os.path.exists(folder_output_path):
    os.makedirs(folder_output_path)

for file_name in os.listdir(folder_input_path):
    file_path = os.path.join(folder_input_path, file_name)
    results = model.predict(source=file_path)

    probs = results[0].probs
    score = round(float(probs.top1conf.cpu().numpy()), 2)
    text = f"{model.names[probs.top1]} {score}"

    img = cv2.imread(file_path)

    font_scale = 2
    thickness = 3  
    img = cv2.putText(img, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 225, 225), thickness)

    output_file_path = os.path.join(folder_output_path, file_name)
    cv2.imwrite(output_file_path, img)

    print(f"Annotated image saved to {output_file_path}")