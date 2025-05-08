from ultralytics import YOLO

model = YOLO("models/3class_v1.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'