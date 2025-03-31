import torch
import cv2
import os

inputDir = "/content/input/"
outputDir = "/content/output/"
microPlasticCnt = 0

model = torch.hub.load('.', 'custom', path='./best.pt', source='local')

# Iterate over files in inputDir
for filename in os.listdir(inputDir):
    filepath = os.path.join(inputDir, filename)
    
    # Check if it's a file
    if os.path.isfile(filepath):
        img = cv2.imread(filepath)
        (b,g,r) = cv2.split(img)
        img1 = b
        binary = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 45)

        # Inference on thresholded image
        results = model(binary)
        coords = []
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf = result[:5]  # Extract x, y coordinates and confidence
            coords.append([(int(x1), int(y1)), (int(x2), int(y2)), conf.item()])  # Append confidence as well
            print(f"Microplastic Count = {microPlasticCnt}, Confidence = {conf.item()}")
            microPlasticCnt += 1

        # Color of the bounding box (BGR format)
        color = (0, 255, 0)  # Green in BGR
        thickness = 2  # Define the thickness of the bounding box outline

        # Draw the bounding box on the image
        for coord in coords:
            cv2.rectangle(img, coord[0], coord[1], color, thickness)
            # Increase font size for confidence display
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.2
            fontColor = color
            lineType = 3
            cv2.putText(img, f'{coord[2]:.2f}', (coord[0][0], coord[0][1] - 10), font, fontScale, fontColor, lineType)

        cv2.imwrite(os.path.join(outputDir, filename), img)
    else:
        print(f"Skipping directory: {filename}")
