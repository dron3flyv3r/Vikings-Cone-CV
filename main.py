import os


from TypeYOLO import TypeYOLO

def main():

    model = TypeYOLO("runs/detect/train3/weights/best.pt")
    model.model.predict(0, stream=True, visualize=True)
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
