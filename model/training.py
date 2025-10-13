from ultralytics import YOLO

if __name__ == "__main__":
    # Load the model (choose one: n, s, m, l)
    model = YOLO("yolo11n.pt")  # smallest, fastest

    # Train the model
    results = model.train(
        data="C:/Users/CHAMA COMPUTERS/Desktop/Data_Science/AI_ML/DeepLearning/Face_Emotion_Detection/Face-Emotion-Detection-YOLOv11/dataset/data.yaml",
        epochs=50,             # adjust as needed
        imgsz=640,
        batch=8,               # reduce if memory error
        workers=0,             # Windows-friendly
        name="glasses-yolo11",
    )

