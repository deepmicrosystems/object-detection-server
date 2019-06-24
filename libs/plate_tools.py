import json
import requests
import scipy.misc
import cv2


class PlateRecognition:
    """
    This has the purpose of detect and read the plates in a image.
    """

    @classmethod
    def get_response_openalrp(cls, image_path):

        """
        Obtain plate information in image_path input cropped image
        :param image_path: Cropped image
        :return: JSON object as dict, if not possible return empty list
        """

        plate_data = {'plate_info': {}}
        try:
            # Obtain response from OpenALP API for image_path input
            response = cls.api_call_plates_openalpr(image_path)
            if len(response['results']) > 0:
                plate_dict = response['results'][0]
                plate = plate_dict['candidates'][0]
                prob = plate_dict['confidence']
                prob = str(round(float(prob) / 100, 2))
                box = plate_dict['coordinates']

                # Set information for plate
                plate_data['plate_info']['plate'] = plate
                plate_data['plate_info']['prob'] = prob
                plate_data['plate_info']['box'] = box
                plate_data['success'] = True
                return plate_data
            else:
                plate_data['success'] = False
                return plate_data
        except Exception as e:
            plate_data['success'] = False
            print(' No internet Connection or: {}', e)
            return plate_data

    @classmethod
    def get_plates(cls, img_path_crop):
        """
        Obtain plates information and append it to dictionary for next process
        :param img_objs: image dict with information of cropped images
        :return: {
            'path': img_path_crop,
            'plate': {'plate': 'NOPLATE'},
            'box': 0,
            'prob': 0
        }
        """

        # Obtain response from API
        read_plate_information = cls.get_response_openalrp( img_path_crop  )

        plate_information = {}

        if read_plate_information['success'] is True:

            plate = read_plate_information['plate_info']['plate']
            prob = read_plate_information['plate_info']['prob']
            box = read_plate_information['plate_info']['box']

            detection = {
                'path': img_path_crop,
                'plate': plate,
                'box': box,
                'prob': prob
            }

            plate_information['detection'] = detection
            plate_information['success'] = True

        else:
            detection = {
                'path': img_path_crop,
                'plate': {'plate': 'NOPLATE'},
                'box': 0,
                'prob': 0
            }

            plate_information['detection'] = detection
            plate_information['success'] = False

        return plate_information

    @staticmethod
    def write_plate(path_to_image='', region=None, plate=''):
        if plate != 'NOPLATE':
            path_to_new_image = path_to_image[:path_to_image.rfind('.')]

            px0 = region[0]['x']
            py0 = region[0]['y']
            px1 = region[2]['x']
            py1 = region[2]['y']

            textx = region[0]['x']
            texty = region[0]['y']

            img = cv2.imread(path_to_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.rectangle(img,(px0,py0),(px1,py1),(0,255,0),3)

            img = cv2.putText(img, plate,
                              (textx, int(texty*0.95)),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1,
                              (0, 255, 3),
                              2,
                              cv2.LINE_AA)
            save_in = "{}_plate.jpg".format(path_to_new_image)
            scipy.misc.imsave(save_in, img)

            return save_in

        else:
            path_to_new_image = path_to_image[:path_to_image.rfind('.')]

            img = cv2.imread(path_to_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_in = "{}_plate.jpg".format(path_to_new_image)
            scipy.misc.imsave(save_in, img)

            return save_in

    @classmethod
    def api_call_plates_openalpr(cls, image):
        """
        This method calls OpenALP API for detect plates in cars, this have limited use calls.
        :param image: Absolute path to image: string.
        :return: JSON response as dict
        """
        # make a prediction on the image
        API_KEY = 'sk_DEMODEMODEMODEMODEMODEMO'
        URL = 'https://api.openalpr.com/v2/recognize?recognize_vehicle=1&country=us&secret_key={}'.format(API_KEY)
        data = {
            'image': open(image, 'rb')
        }

        headers = {
            'accept': 'multipart/form-data'
        }

        response = requests.post(URL, headers=headers,
                                 files=data,
                                 auth=requests.auth.HTTPBasicAuth(API_KEY, ''))
        return json.loads(response.text)

if __name__ == '__main__':
    pass