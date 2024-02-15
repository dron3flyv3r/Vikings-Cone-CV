
# This script downloads the dataset from Roboflow.
import os
from dotenv import load_dotenv
load_dotenv()

from roboflow import Roboflow
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("automated-historical-document-processing-fei-stu-in-bratislava").project("fsoco-94q5z")
dataset = project.version(1).download("yolov8")
