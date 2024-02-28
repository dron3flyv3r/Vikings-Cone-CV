import numpy as np
import onnxruntime as ort
import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image):
        # Resize and normalization steps here depend on your model's requirements
        # This is a generic preprocessing step, adjust as necessary
        image_resized = cv2.resize(image, (640, 640))  # Example resize, adjust to your model's input size
        image_normalized = image_resized / 255.0  # Normalize to [0, 1] if required by your model
        image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Change data layout to CHW if required
        image_batch = np.expand_dims(image_transposed, axis=0).astype(np.float32)
        return image_batch

    def predict(self, image):
        # Preprocess the image
        input_data = self.preprocess(image)

        # Run the model
        output_tensor = self.session.run(self.output_names, {self.input_name: input_data})[0]

        # Reshape the output tensor for easier manipulation
        output_tensor = output_tensor.reshape(-1, 9)  # Assuming the model outputs in [1, 9, 8400] format

        results = []
        for detection in output_tensor:
            # Assuming the first 5 elements are [x, y, width, height, confidence]
            x, y, width, height, raw_confidence = detection[:5]
            class_probabilities = detection[5:]
            
            # Normalize or adjust the confidence score if necessary
            # This step depends on how your model represents confidence scores
            # Example: confidence = sigmoid(raw_confidence) if raw_confidence is not in [0, 1]
            confidence = 1 / (1 + np.exp(-raw_confidence))  # Example sigmoid function, adjust as necessary

            # Determine the class with the highest probability
            class_id = np.argmax(class_probabilities)
            class_confidence = class_probabilities[class_id]

            # Combine objectness score with class confidence if necessary
            overall_confidence = confidence * class_confidence  # Adjust based on your model's output format

            # Apply threshold to filter out weak detections
            if overall_confidence > 0.5:  # Adjust threshold as necessary
                box = [x, y, width, height]
                result = {
                    'class_id': class_id,
                    'confidence': overall_confidence,
                    'box': box,
                }
                results.append(result)

        return results

# model = YOLODetector("runs/detect/train3/weights/best.onnx")
img = cv2.imread("test/4meter1.jpg")
# model = YOLO("runs/detect/train3/weights/best.onnx")
# output = model.predict(img, device="cpu")

from TypeYOLO import TypeYOLO
# model1 = TypeYOLO("runs/detect/train3/weights/best.pt")
# model1.model.export(format="OpenVINO", half=True, simplify=True)
model = TypeYOLO("runs/detect/train3/weights/best.onnx")

# import timeit

# def timeModel(model_path: str, img_path: str) -> float:
#     print("Timing model type:", model_path.split(".")[-1])
#     global model
#     times = 1000
#     model = TypeYOLO(model_path)
#     start = timeit.timeit(
#         f"model.predict('{img_path}')",
#         "from __main__ import model",
#         number=times
#     )
#     return start


# onnx_time = timeModel("runs/detect/train3/weights/best.onnx", r"test/4meter1.jpg")
# pt_time = timeModel("runs/detect/train3/weights/best.pt", r"test/4meter1.jpg")

# print("TOTAL TIME")
# print(f"ONNX: {onnx_time} PT: {pt_time}")
# print("AVERAGE TIME")
# print(f"ONNX: {onnx_time/1000} PT: {pt_time/1000}")


model.predict(1)
exit()
output = model.predict(img)
print("Teting full model")
output = model.predict(img)
output = model.predict(img)
output = model.predict(img)
output = model.predict(img)
print("Teting half model")
output = model.predict(img, half=True)
output = model.predict(img, half=True)
output = model.predict(img, half=True)
output = model.predict(img, half=True)

# print("Loading pt model...")
# model = TypeYOLO("runs/detect/train3/weights/best.pt")
# output = model.predict(img)
# output = model.predict(img)
# output = model.predict(img)
# output = model.predict(img)
# output = model.predict(img)
# output = model.predict(img)

# print(len(output))
# for o in output:
#     print(o)


# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO("runs/detect/train/weights/best.pt")

# cap = cv2.VideoCapture(0)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
