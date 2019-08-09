import sqlite3
import time 
import datetime

class DataBaseManager:
    def __init__(self, db_name = None):
        self.conn = None
        self.cursor = None
        
        if db_name:
            self.open(db_name)

    def open(self, db_name):
        try:
            self.conn = sqlite3.connect(db_name)
            self.cursor = self.conn.cursor()
            self.create_table()

        except Exception as e:
            print('Cannot connect to db or {}'.format(e))

    def create_table(self, case):
        if (case == "detections"):
            self.cursor.execute("CREATE TABLE IF NOT EXISTS \
                         detections( item_id REAL, w REAL, h REAL, x REAL, y REAL, prob REAL,\
                                     datestamp TEXT, class TEXT, imgPath TEXT)") 
        elif (case == "plates"):
            self.cursor.execute("CREATE TABLE IF NOT EXISTS \
                         plates( item_id REAL, w REAL, h REAL, x REAL, y REAL, prob REAL,\
                                     datestamp TEXT, imgPath TEXT)")   
    def close(self):
        self.cursor.close()
        self.conn.close()

    def __enter__(self):
        
        return self

    # def __exit__(self,exc_type,exc_value,traceback):
    #     self.close()

    def dynamic_data_entry(self, item_id, image_path, detection, prob, obj_class, date):

        x = detection["xmin"]
        y = detection["ymin"]
        h = detection["xmax"]
        w = detection["ymax"]

        self.cursor.execute("INSERT INTO detections \
                        (item_id, w, h, x, y, prob, datestamp, class, imgPath) \
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (item_id, w, h, x, y, prob, date, obj_class, image_path))
        self.conn.commit()
        #self.close()

    def dynamic_data_entry_plates(self,image_path_crop, detection, plate, prob , date,  item_id):
        
        x = detection["xmin"]
        y = detection["ymin"]
        h = detection["xmax"]
        w = detection["ymax"]

        self.cursor.execute("INSERT INTO plates \
                        (item_id, w, h, x, y, prob, datestamp, imgPath) \
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (item_id, w, h, x, y, prob, date, image_path_crop))

        self.conn.commit()
        #self.close()
