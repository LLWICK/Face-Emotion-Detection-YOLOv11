from ultralytics import YOLO
import cv2

model = YOLO("../runs/detect/glasses-yolo112/weights/best.pt")
results = model.predict(
    "dataset/test/images/9k_-9-_face_png.rf.2dd6cdaee4cad517228b42a3df456fc8.jpg",
    show=True,
    save = True
)




