from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("C:/Users/CHAMA COMPUTERS/Desktop/Data_Science/AI_ML/DeepLearning/Face_Emotion_Detection/Face-Emotion-Detection-YOLOv11/runs/detect/glasses-yolo112/weights/best.pt")  # or your custom trained model

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, device=0)  # results is a list

    # Draw the detections on the frame
    annotated_frame = results[0].plot()  # take the first result

    # Display
    cv2.imshow("YOLO Detection", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
