from time import time
from ultralytics import YOLOv10, YOLO
from ultralytics.engine.results import Results
from typing import Literal

class ConePosition:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, objectType: Literal["yellow", "blue", "orange", "unknown"], results: Results, conf: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.type: Literal["yellow", "blue", "orange", "unknown"] = objectType
        self.results = results
        self.conf = conf
        
    def plot(self):
        self.results.plot(show=True)
        
    def save(self, path: str):
        self.results.plot(save=True, filename=path)
        
    def __str__(self):
        text = "<Cone: "
        for key, value in self.__dict__.items():
            if key == "results":
                continue
            text += f"{key}={value},  "
        text = text[:-3]
        text += ">"
        return text
    
    def __repr__(self):
        return str(self)
    
class Cones:
    def __init__(self, cones: list[ConePosition], img_width: int, img_height: int, time: float):
        self.cones = cones
        self.img_width = img_width
        self.img_height = img_height
        self.time = time
        self.num_cones = len(cones)
        self.fps = 1 / time
        self.img_size = (img_width, img_height)
        
    def plot(self):
        if len(self.cones) > 0:
            self.cones[0].plot()
            
    def save(self, path: str):
        if len(self.cones) > 0:
            self.cones[0].save(path)
    
    def __str__(self):
        text = "<Cones: "
        for key, value in self.__dict__.items():
            text += f"{key}={value}, "
        text = text[:-2]
        text += ">"
        return text
    
    def __repr__(self):
        return str(self)

class BetterYoloV10:
    model: None | YOLOv10
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def _setup_model(self) -> YOLOv10:
        model: YOLOv10 = YOLOv10(self.model_path, task="detect", verbose=False)# type: ignore
        # print("Model loaded on", model.device)
        return model
    
    def manual_setup(self, path: str, v10: bool = True) -> None:
        self.model_path = path
        self.model = None
        
        if v10:
            self.model = self._setup_model()
        else:
            self.model = YOLO(path, task="detect", verbose=False) # type: ignore
        
    def raw_detect(self, img, conf=0.25) -> list[Results]:
        
        if self.model is None:
            self.model = self._setup_model()
        
        results = self.model.predict(
            img,
            conf=conf,
            max_det=6,
        )
        
        return results
    
    def detect(self, img, conf=0.25) -> Cones:
        
        # - blue_cone
        # - large_orange_cone
        # - orange_cone
        # - unknown_cone
        # - yellow_cone
        
        start_time = time()
        
        idxToCone: list = [
            "Blue",
            "Orange", # Large Orange - Just the same as Orange Cone
            "Orange",
            "Unknown",
            "Yellow"
        ]
        
        results = self.raw_detect(img, conf)
        
        width = results[0].orig_shape[1]
        height = results[0].orig_shape[0]
        
        cones = []
        
        for result in results:
            for b in result.boxes or []:
                x1, y1, x2, y2 = tuple(b.xyxy.tolist()[0])
                
                x1 /= width
                x2 /= width
                y1 /= height
                y2 /= height
                
                cones.append(ConePosition(x1, y1, x2, y2, idxToCone[int(b.cls.item())], result, b.conf.item()))
        
        return Cones(cones, width, height, time() - start_time)
    
    def compile(self, device: Literal["cpu", "gpu"] = "cpu") -> None:
        
        if self.model is None:
            self.model = self._setup_model()
        
        # Export the model and set the correct path
        self.model.export(format="onnx", simplify=True)
        
        # TODO: Need to make for a raspberry pi and windows
    
        
        # # check if os is windows
        # if os.name == "nt":
        #     if self.model.device == "cpu":
        #         self.model.export(format="openvino", half=True)
        #     else:
        #         self.model.export(format="openvino")
        # else:
        #     self.model.export(format="ncnn")
        
    
    