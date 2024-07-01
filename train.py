######## This is so the terminal can run without userwarnings ########
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
######################################################################

from ultralytics import YOLOv10
from ultralytics.engine.results import Results
# from ultralytics.models.yolov10.model import YOLOv10


# Change the dataset_path to the path of the working directory

def main():
    # Initialize model
    model: YOLOv10 = YOLOv10("runs/detect/train5/weights/best.onnx") # type: ignore
    
    print(type(model))
    
    ########################################################################################
    # yolov8n: 640 Img Size 3.2M Params     8.7B FLOPs    (Fastest, lowest accuracy)       #
    # yolov8s: 640 Img Size 11.2M Params    28.6B FLOPs   (Fast, low accuracy)             #
    # yolov8m: 640 Img Size 25.9M Param     78.9B FLOPs   (Moderate, moderate accuracy)    #
    # yolov8l: 640 Img Size 43.7M Params    165.2B FLOPs  (Slow, high accuracy)            #
    # yolov8x: 640 Img Size 68.2M Params    257.8B FLOPs  (Slowest, highest accuracy)      #
    ########################################################################################

    ##### Hyperparameters #####
    parameters = {
        "imgsz": 640, # Dont change this value unless you change the model
        "batch": 32, # Good idea to lower the batch size if you run out of memory
        # "epochs": 100, # Number of epochs to train the model, 100 is a good starting point
        "time": 5, # Number of hours to train the model, 5 is a good starting point
        "data": "merged-data/data.yaml", # Path to the data.yaml file
        "device": "0", # Change this to 0 if you have a GPU or "cpu" if you dont have a GPU
        "dropout": 0.1, # Dropout rate - 0.1 is a good starting point for YOLO
        "save_period": 10, # Save model every 10 epochs
    }
    
    ###### Train model #####
    # model.train(**parameters)

    #yolo task=detect mode=train batch=32 plots=True model=yolov10n.pt data=merged-data/data.yaml device=0 time=5 dropout=0.1 save_period=10
    
    # Q: how do I install huggingface_hub? 
    # A: pip install huggingface_hub
    
    # results: list[Results] = model("test/4meter1.jpg", conf=0.25)
    # import time
    # import cv2
    # import torchvision
    # from torchvision.transforms import Resize
    
    # img = torchvision.io.read_image("test/4meter1.jpg")
    # img = Resize((640, 640))(img)
    # img = img.unsqueeze(0)
    # # Save the image
    # torchvision.io.write_jpeg(img[0], "test/4meter1_resized.jpg")
    # exit()
    
    #Print all the input arguments that the model __call__ function takes
    # print(model.__call__.__code__.co_varnames)
    # exit()
    
    # img = cv2.imread("test/4meter1_resized.jpg")
    
    # start = time.time()
    # for _ in range(100):
    #     model(img, conf=0.25, device="cpu") # type: ignore
    # end = time.time()
    # print("Time taken: ", end-start)
    # print("FPS: ", 100/(end-start))
    # print("Avg time per image: ", (end-start)/100)
    
    
    

if __name__ == "__main__":
    main()
    