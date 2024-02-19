import threading
from typing import Generator
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import threading

class TypeYOLO:
    
    def __init__(self, model: str, task: str | None = None, verbose: bool = False) -> None:
        """
        Initializes a YOLOModel object.

        Args:
            model (str): The path to the YOLO model file.
            task (str | None, optional): The task to perform using the model. Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
        """
        self.model = YOLO(model, task, verbose)
        self._open_window = False
        self._stop_thread = False
        self._current_image = None
        
    def train(self, data_path: str, imgsz: int = 640, batch: int = 32, epochs: int = 100, device: str = "0", dropout: float = 0.1, save_period: int = 10) -> None:
            """
            Trains the YOLO model using the specified parameters.

            Args:
                data_path (str): The path to the training data.
                imgsz (int, optional): The input image size. Defaults to 640.
                batch (int, optional): The batch size. Defaults to 32.
                epochs (int, optional): The number of training epochs. Defaults to 100.
                device (str, optional): The device to use for training. Defaults to "0".
                dropout (float, optional): The dropout rate. Defaults to 0.1.
                save_period (int, optional): The period at which to save the model. Defaults to 10.

            Returns:
                None
            """
            parameters = {
                "imgsz": imgsz,
                "batch": batch,
                "epochs": epochs,
                "data": data_path,
                "device": device,
                "dropout": dropout,
                "save_period": save_period,
            }
            self.model.train(**parameters)
        
    def predict(self, *images: list[str] | str, show: bool = False, min_conf: float = 0.5) -> list[Results]:
        """
        Predicts the results for the given images.

        Args:
            images (list[str] | str): A list of image paths or a single image path.
            show (bool, optional): Whether to display the results on a window. Defaults to False.
            continueastion (bool, optional): Whether to continue have the window open for the next predictions. Defaults to False.
        
        Returns:
            list[Results]: A list of Results objects representing the predictions.
        
        Raises:
            TypeError: If the model is unable to return a Results object.
        """
        # if isinstance(images, str):
        #     images = [images]
        results: list[Results] = []
        for img in images:
            out = self.model.predict(img, conf=min_conf)
            results.extend(out)
        
        
        if isinstance(results, Results):
            results = [results]
        elif isinstance(results, list):
            results = results
        else:
            raise TypeError("Sorry, but the model is unable to return a 'Results' object.")
        
        if show:
            self.show(results)
        
        return results
    
    def stream(self, source: str | list[str], *, show: bool = False, block: bool = False) -> Generator[Results, None, None]:
            """
            Streams frames from a video source or a list of image paths and applies the YOLOv8 model to each frame.

            Args:
                source (str | list[str]): The video source or a list of image paths.
                show (bool, optional): Whether to display the results on a window. Defaults to False.

            Yields:
                Results: The results of applying the YOLOv8 model to each frame.

            Raises:
                TypeError: If the model is unable to return a 'Results' object.

            """
            self._create_window("YOLOv8")
            
            if isinstance(source, list):
                for scr in source:
                    results: Results = self.model(scr)
                    if isinstance(results, Results):
                        if show:
                            self._update_window("YOLOv8", results)
                        yield results
                    else:
                        raise TypeError("Sorry, but the model is unable to return a 'Results' object.")
            else:
                cap = cv2.VideoCapture(source)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results: Results = self.model(frame)[0]
                    if isinstance(results, Results):
                        if show:
                            self._update_window("YOLOv8", results)
                        yield results
                    elif isinstance(results, list):
                        if show:
                            self._update_window("YOLOv8", result)
                        for result in results:
                            yield result
                    else:
                        raise TypeError("Sorry, but the model is unable to return a 'Results' object.")
                cap.release()
                
            if block:
                cv2.waitKey(0)
            self._destroy_all_windows()
        
    def show(self, result: Results | list[Results]) -> None:
            """
            Display the results using the provided Results object.

            Args:
                result (Results): The Results object containing the results to be displayed.

            Returns:
                None
            """
            self._create_window("YOLOv8")
            self._update_window("YOLOv8", result)
            
    def show_comparison(self, comp1: Results | list[Results], comp2: Results | list[Results], comp1_title: str = "Model A", comp2_title: str = "Model B") -> None:
        """
        Display the comparison results using the provided Results objects.

        Args:
            comp1 (Results | list[Results]): The first set of results to be displayed.
            comp2 (Results | list[Results]): The second set of results to be displayed.
            comp1_title (str): Title for the first comparison window. Default is "Model A".
            comp2_title (str): Title for the second comparison window. Default is "Model B".

        Returns:
            None
        """
        self._create_window(comp1_title)
        self._create_window(comp2_title)
        
        # Update both windows before waiting for a key press
        self._update_window(comp1_title, comp1)
        self._update_window(comp2_title, comp2)
        
        # Wait for a key press to close both windows
        cv2.waitKey(0)
        self._destroy_window(comp1_title)
        self._destroy_window(comp2_title)

        
    def _create_window(self, window_name: str) -> None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
    def _destroy_window(self, window_name: str) -> None:
        cv2.destroyWindow(window_name)
        
    def _destroy_all_windows(self) -> None:
        cv2.destroyAllWindows()
        
    def _convert_to_grid(self, results: list[Results]) -> np.ndarray:
        
        # Calculate the number of rows and columns
        n = len(results)
        rows = int(np.sqrt(n))
        cols = n // rows
        
        images = [result.plot() for result in results]
        
        # Create a list of rows with the images in each row
        rows = [np.hstack(images[i:i + cols]) for i in range(0, n, cols)]
        
        # Combine the rows into a single image
        return np.vstack(rows)
                        
    def _update_window(self, window_name: str, result: Results | list[Results]) -> None:
        
        if isinstance(result, list):
            result = self._convert_to_grid(result)
        else:
            result = result.plot()
            
        cv2.imshow(window_name, result)
        cv2.waitKey(0)
        
    def save(self, result: Results, path_name: str = "result.jpg") -> None:
        """
        Saves the result image.

        Args:
            result (Results): The result image to be saved.
            path_name (str, optional): The path and name of the saved image. Defaults to "result.jpg".
        """
        result.save(filename=path_name)
        
    def save_all(self, results: list[Results], path: str) -> None:
        """
        Save all the results as images.

        Args:
            results (list[Results]): The list of results to be saved.
            path (str): The path where the images will be saved.

        Returns:
            None
        """
        for i, result in enumerate(results):
            result.save(filename=f"{path}/result_{i}.jpg")
            
    def __call__(self, *images: list[str] | str, show: bool = False) -> list[Results]:
        return self.predict(*images, show=show)
        