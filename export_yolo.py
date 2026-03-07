from ultralytics import YOLO
import shutil
import os

print("Loading YOLOv8n base model...")
model = YOLO("yolov8n.pt")

print("Exporting model to ONNX with opset 17...")
model.export(format="onnx", opset=17)

src_onnx = "yolov8n.onnx"
dest_onnx = "models/vision/yolov26n_agri_defects.onnx"

print(f"Moving {src_onnx} to {dest_onnx}...")
os.makedirs("models/vision", exist_ok=True)
shutil.move(src_onnx, dest_onnx)
print("Done!")
