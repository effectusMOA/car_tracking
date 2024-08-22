import cv2
from ultralytics import RTDETR
import math
# Load the YOLOv8 model
model = RTDETR("rtdetr-l.pt")

# Open the video file
video_path = "bus.mp4"
mask = cv2.imread("bus_mask.png")
cap = cv2.VideoCapture(video_path)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign"
              
              ]


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame,mask)

    if success:
        # Run YOLOv8 inference on the frame
        results = model(imgRegion)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1),int(x2),int(y2)
                w, h = x2- x1, y2- y1
                conf = math.ceil((box.conf[0]*100)) / 100
                cls = int(box.cls[0])
                
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw the circle at the center of the bounding box

                if conf > 0.5:# Put the class name at the top-left corner of the bounding box
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    label = f"{classNames[cls]} {conf}"
                    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
                

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("RTDETR Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()