######## This is so the terminal can run without userwarnings ########
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
######################################################################

from ultralytics import YOLO


# Change the dataset_path to the path of the working directory

def main():
    # Initialize model
    model = YOLO("yolov8n.pt")
    
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
        "epochs": 100, # Number of epochs to train the model, 100 is a good starting point
        "data": "merged-data/data.yaml", 
        "device": "0", # Change this to 0 if you have a GPU or "cpu" if you dont have a GPU
        "dropout": 0.1, # Dropout rate - 0.1 is a good starting point for YOLO
        "save_period": 10, # Save model every 10 epochs
    }
    
    ###### Train model #####
    model.train(**parameters)

if __name__ == "__main__":
    main()
    