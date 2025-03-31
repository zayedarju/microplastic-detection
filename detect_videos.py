import torch
import cv2
import os

input_video_path = "/content/input/input_video.mp4"
output_video_path = "/content/output/output_video.mp4"
microPlasticCnt = 0

model = torch.hub.load('.', 'custom', path='./best.pt', source='local')

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Apply thresholding
    (b, g, r) = cv2.split(frame)
    img1 = b
    binary = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 45)

    # Inference on thresholded frame
    results = model(binary)
    if results is None:
        print(f"Warning: No results for frame {frame_count}")
        continue

    coords = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf = result[:5]  # Extract x, y coordinates and confidence
        coords.append([(int(x1), int(y1)), (int(x2), int(y2)), conf.item()])  # Append confidence as well
        microPlasticCnt += 1
        print(f"Microplastic Count = {microPlasticCnt}, Confidence = {conf.item()}")

    # Color of the bounding box (BGR format)
    color = (0, 255, 0)  # Green in BGR
    # Define the thickness of the bounding box outline
    thickness = 2

    # Draw the bounding boxes on the original frame
    for coord in coords:
        cv2.rectangle(frame, coord[0], coord[1], color, thickness)
        # Increase font size for confidence display
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.0
        fontColor = color
        lineType = 2
        cv2.putText(frame, f'{coord[2]:.2f}', (coord[0][0], coord[0][1] - 10), font, fontScale, fontColor, lineType)

    # Write the frame into the output video
    out.write(frame)

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to:", output_video_path)
