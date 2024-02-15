!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="soAFgH73zoJNZK9sLM5e")
project = rf.workspace("automated-historical-document-processing-fei-stu-in-bratislava").project("fsoco-94q5z")
dataset = project.version(1).download("yolov8")
