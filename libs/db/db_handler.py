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
            print(f'Cannot connect to db or {e}')

    def create_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS \
                         detections( w REAL, h REAL, x REAL, y REAL, prob REAL,\
                                     datestamp TEXT, class TEXT, imgPath TEXT)")   
    def close(self):
        self.cursor.close()
        self.conn.close()

    def __enter__(self):
        
        return self

    # def __exit__(self,exc_type,exc_value,traceback):
    #     self.close()

    def dynamic_data_entry(self, image_path, detection, prob, obj_class, date):

        x = detection["xmin"]
        y = detection["ymin"]
        h = detection["xmax"]
        w = detection["ymax"]

        self.cursor.execute("INSERT INTO detections \
                        (w, h, x, y, prob, datestamp, class, imgPath) \
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (w, h, x, y, prob, date, obj_class, image_path))

        self.conn.commit()
        #self.close()
