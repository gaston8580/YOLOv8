from ultralytics import YOLO

model = YOLO('yolov8s.yaml')  # load a pretrained model
model.train(data='yolo_data.yaml', epochs=100, batch=64)  # train the model