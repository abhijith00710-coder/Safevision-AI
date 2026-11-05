from ultralytics import YOLO

# Load pretrained YOLO26 nano model
model = YOLO("yolo26n.pt")

# Train on Construction-PPE dataset
results = model.train(
    data="construction-ppe.yaml",
    epochs=5,
    imgsz=640,
    device="cpu"
)

print("Training completed!")