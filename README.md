## AI Vistion (Vikings)


### Setup Instructions
1. Clone the repository
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
3. You well need to manual install the following packages to use the YoloV10 Model:
```bash
pip install -q git+https://github.com/THU-MIG/yolov10.git
```

### How to use the code
The first thing you need to do is to import the required packages:
```python
from BetterYoloV10 import BetterYoloV10
```
Then you need to create an instance of the BetterYoloV10 class:
```python
yolo = BetterYoloV10("path/to/weights.pth")
```
Then you can start 'detecting' cones in an image using the following code:
```python
image_path = "path/to/image.jpg"
result = yolo.detect(image_path)
```
The return is a simple object with all nessesary information:
- results.cones: This is a list of all the cones detected in the image. Each cone is another simple object (Read about it below).
- results.num_cones: This is the number of cones detected in the image.
- results.img_width: This is the width of the image.
- results.img_height: This is the height of the image.
- results.time: This is the time taken to detect the cones in the image.
- results.fps: This is the frames per second of the detection.
- results.img_size: This is a tuple of the width and height of the image.

Methods:
- results.plot() : This method will show the image with the cones detected.
- results.save("path/to/save.jpg") : This method will save the image with the cones detected.

Each cone object has the following attributes:
- cone.x1: This is the x coordinate of the top left corner of the cone.
- cone.y1: This is the y coordinate of the top left corner of the cone.
- cone.x2: This is the x coordinate of the bottom right corner of the cone.
- cone.y2: This is the y coordinate of the bottom right corner of the cone.
- cone.conf: This is the confidence of the detection.
- cone.type: This is the type of the cone. It can be either "yellow", "blue", "orange" or "unknown"
- cone.results: This is the raw data from the model. It is a special format from the ultralytics.engine

You can also print any of the objects to see the data in a human readable format.

### Example
```python
from BetterYoloV10 import BetterYoloV10

yolo = BetterYoloV10("path/to/weights.pth")
result = yolo.detect("path/to/image.jpg")

print("Number of cones detected:", result.num_cones)
print("Time taken to detect cones:", result.time)
print("Frames per second:", result.fps)

for c in results.cones:
    print(c)

results.plot()
results.save("path/to/save.jpg")
```