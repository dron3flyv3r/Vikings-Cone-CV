from BetterYoloV10 import BetterYoloV10


# First we create the yolo object - It need the path to the YoloV10!! Model this can't use the YoloV8 model
yolo = BetterYoloV10(
    model_path="runs/detect/train5/weights/best.onnx",
)

# If you want to use the YoloV8 model you can use the manual_setup method
# yolo.manual_setup(
#     path="<path to the YoloV8 model>", # Path to the YoloV8 model
#     v10=False, # This is important to set to False - If you don't set this to False it will try and use the YoloV10 model (And get an error)
# )

# Now we can detect cones in images
image_path = "test/4meter1.jpg"
results = yolo.detect(
    image_path, # Path to the image
    conf=0.5, # Confidence threshold - This is the minimum confidence for a cone to be detected
)

# In the results object we have some useful information
results.cones
results.num_cones


# It has also some useful methods to use while debugging or testing
results.plot() # This will show the image with the detected cones
results.save("output/4meter1_v10.jpg") # This will save the image with the detected cones

# In the results.cones we have a list of cone objects
cones = results.cones

# This objects have some useful information 
cone = cones[0]
cone.type # This is the type of the cone - blue, yellow, orange or unknown
cone.x1 # This is the x1 coordinate of the cone in the image - it is normalized between 0 and 1 (0 is the left side of the image and 1 is the right side)
cone.y1 # This is the y1 coordinate of the cone in the image - it is normalized between 0 and 1 (0 is the top side of the image and 1 is the bottom side)
cone.x2 # This is the x2 coordinate of the cone in the image - it is normalized between 0 and 1 (0 is the left side of the image and 1 is the right side)
cone.y2 # This is the y2 coordinate of the cone in the image - it is normalized between 0 and 1 (0 is the top side of the image and 1 is the bottom side)
cone.results # This is the raw results from the model - This is a list of boxes with the detected cones

# We can also print the results
print(results) # <Cones: <num_of_cones> cones detected in <time_it_took> seconds at <estimated_FPS> FPS>
print(cone) # <type> cone at (<x1>, <y1>) to (<x2>, <y2>)