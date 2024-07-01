import os

from TypeYOLO import TypeYOLO
from BetterYoloV10 import BetterYoloV10

def main():
    yolo = BetterYoloV10("runs/detect/train5/weights/best.onnx")
    # yolo.compile()
    # yolo.model.export(format="ncnn", half=True, imgsz=640)
    
    results = yolo.detect("test/4meter1.jpg")
    print(results)
    # results = yolo.detect("test/4meter1.jpg")
    # results.save("output/4meter1_v10.jpg")
    
    # results = yolo.detect("test/10meter3.jpg")
    # results.save("output/10meter3_v10.jpg")
    
    # yolo.manual_setup("runs/detect/train3/weights/best.onnx", False)
    # results = yolo.detect("test/4meter1.jpg")
    # results.save("output/4meter1_v8.jpg")
    
    # results = yolo.detect("test/10meter3.jpg")
    # results.save("output/10meter3_v8.jpg")
        
    
    # print(results)
    # results = yolo.detect("test/4meter1.jpg")
    # print(results)
    # results = yolo.detect("test/4meter1.jpg")
    # print(results) 
    
    # for r in results:
    #     print(r)
    
    # for result in results:
    #     for b in result.boxes or []:
    #         print("Cones:", b.cls.item())
    #         print("Confidence:", b.conf.item())
    #         print("Coordinates:", tuple(b.xyxy.tolist()[0]))
            
    
    

def oldMain():

    model = TypeYOLO("runs/detect/train2/weights/best.pt")
    # model.model.export(format="OpenVINO", half=True, simplify=True, device=0)
    # and onnx
    # model.model.export(format="ONNX", half=True, simplify=True, device=0)
    folder_path = "output/"
    file_name = "goproVideo{idx}.png"
    list_of_files = os.listdir("D:/temp/output_dir")[500:1000]
    print(len(list_of_files))
    list_of_files = ["D:/temp/output_dir/" + file for file in list_of_files]
    for idx, o in enumerate(model.model.predict(list_of_files, stream=True, device=0)):
        o.save(folder_path + file_name.format(idx=idx))
    # model.stream(1, show=True)

    return

    # Create a model
    modelA = TypeYOLO("runs/detect/train2/weights/best.pt")
    modelB = TypeYOLO("runs/detect/train3/weights/best.pt")

    # Putting images into a list
    test_images_path = "test"
    images = [f"{test_images_path}/{i}" for i in os.listdir(test_images_path)]

    # Run the model on the images and save the results
    resultsA = modelA(images)
    resultsB = modelB(images)

    for A, B in zip(resultsA, resultsB):
        confA = [f"{i:.4f}" for i in A.boxes.conf.tolist()]
        confB = [f"{i:.4f}" for i in B.boxes.conf.tolist()]
        print(f"Model A CONF: {confA} Model B CONF: {confB}")

    modelA.show_comparison(resultsA, resultsB, "Train 1 Model", "Train 2 Model")
    # Show the results
    # for rA, rB in zip(resultsA, resultsB):
    #   modelA.show_comparison(rA, rB, "Model A", "Model B")


if __name__ == "__main__":
    main()
