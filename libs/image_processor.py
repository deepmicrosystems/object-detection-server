from .plate_tools import PlateRecognition
import uuid
from .tools import *

class ImageProcessor:
    def __init__(self):
        self.predictions = None
        self.db = None
        self.plateRecognition = PlateRecognition()
        
        self.db_state = None
        self.db_state_plate = None

    def start_modules(self, db, db_path):
        self.db = db
        try:
            self.db.open(db_name=db_path)
            return True
        except:
            return False
    
    @staticmethod
    def save_in_db(db, detection, image_path, date, item_id):
        finish=False
        try:
            db.dynamic_data_entry(image_path = image_path,
                                    detection = detection["coord"], 
                                    obj_class=detection["class"],
                                    prob = detection["probability"],
                                    date = date,
                                    item_id = item_id
                                )
            finish = True
        except Exception as e:
            print(f'[INFO ERROR] Cannot sasave_in_dbve db for image {image_path} or {e}')
        
        return finish
    
    @staticmethod
    def save_in_db_plates(db, image_path_crop, date, item_id, plate_info):
        finish=False
        try:
            db.dynamic_data_entry_plates(image_path_crop=image_path_crop,
                                    detection= plate_info["box"], 
                                    plate = plate_info["plate"],
                                    prob = plate_info["prob"],
                                    date=date,
                                    item_id = item_id
                                )
            finish = True
        except Exception as e:
            print(f'[INFO ERROR] Cannot save db for image {image_path_crop} or {e}')
        
        return finish


    def image_saver(self, frame, path_to_img):
        """
        return: tuple (image_path String, timestamp String)
        """
        unix = int(time.time())
        date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))

        image_path = path_to_img + f"{date}.jpg"

        print('IMAGE PATH is ' + image_path)
        
        cv2.imwrite(image_path, frame)

        return image_path, date

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

    def __call__(self, db_path, detections, frame, path_to_img, item_id):

        image_path, date = self.image_saver( frame, path_to_img)
        
        self.db.open(db_name=db_path)

        for d in detections["predictions"]:

            image_path_crop = self.cut_object(d, image_path)
            
            self.db_state = self.save_in_db(self.db, 
                            d, 
                            image_path, 
                            date, 
                            item_id)

            # Obtain plate
            plate_information = self.plateRecognition.get_plates(image_path_crop)

            print('plate information', plate_information)

            if (plate_information["success"]):
               self.db_state_plate = self.save_in_db_plates(self.db,    
                                                    image_path_crop,
                                                    date,
                                                    item_id,
                                                    plate_information)


class Plate:

    plate = str

    x = int
    y = int
    w = int
    h = int

    image_path = str