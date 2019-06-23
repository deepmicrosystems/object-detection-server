from .plate_tools import PlateRecognition
import uuid
from .tools import *

class ImageProcessor:
    def __init__(self):
        self.predictions = None
        self.db = None
        self.plateRecognition = PlateRecognition()


    def start_modules(self, db, predictions):
        self.db = db
        self.predictions = predictions
    
    @staticmethod
    def save_in_db(db, detection, image_path,date):

        save_in_db(db, detection, image_path, date)

    def image_saver(self, frame, path_to_img):
        return image_saver( frame, path_to_img)

    def cut_object(self, detections, image_path):
        x = detections["coord"]["xmin"]
        y = detections["coord"]["ymin"]
        w = detections["coord"]["xmax"]
        h = detections["coord"]["ymax"]

        image = cv2.imread(image_path)

        image_crop = image[y: y+h, x:x + w]
        
        image_path_crop =  "/tmp/" +  str(uuid.uuid4()) + ".jpg"

        cv2.imwrite(image_path_crop, image_crop)
        
        return image_path_crop

    def __call__(self, db_path, detections, frame, path_to_img):

        image_path, date = self.image_saver( frame, path_to_img)
        
        self.db.open(db_name=db_path)
        for d in detections["predictions"]:

            image_path_crop = self.cut_object(d, image_path)

            find_plate = self.plateRecognition.get_response_openalrp(image_path_crop)

            if (find_plate):
                self.save_in_db = save_in_db(self.db, d, image_path, date)
