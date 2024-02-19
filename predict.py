import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from TypeYOLO import TypeYOLO


yolo = TypeYOLO("runs/detect/train/weights/best.pt")


test_path = "test"
images = [os.path.join(test_path, img) for img in os.listdir(test_path) if "result" not in img]
    
results = yolo.predict(*images, show=True)