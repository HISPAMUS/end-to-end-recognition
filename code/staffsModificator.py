import matplotlib.pyplot as plt
import cv2
import json
import random
import numpy as np

class StaffsModificator:
    __params = dict()
    __params['pad'] = 0.1
    
    # Contrast
    __params['clipLimit'] = 1.0

    # Erosion and Dilation
    __params['kernel'] = 4

    def __init__(self, lst_rute, **options):
        self.__params['rotation_rank'] = options['rotation'] if options.get("rotation") else 0
        self.__params['random_margin'] = options['margin'] if options.get("margin") else 0
        self.__params['erosion_dilation'] = options['erosion_dilation'] if options.get("erosion_dilation") else False
        self.__params['dilation'] = options['dilation'] if options.get("dilation") else False
        self.__params['iterations'] = options['iterations'] if options.get("iterations") else 1
        self.__params['lst_rute'] = lst_rute

    def __rotate_point(self, M, center, point):
        point[0] -= center[0]
        point[1] -= center[1]

        point = np.dot(point, M)

        point[0] += center[0]
        point[1] += center[1]

        return point

    def __rotate_points(self, M, center, top, bottom, left, right):
        left_top     = self.__rotate_point(M, center, [left, top])
        right_top    = self.__rotate_point(M, center, [right, top])
        left_bottom  = self.__rotate_point(M, center, [left, bottom])
        right_bottom = self.__rotate_point(M, center, [right, bottom])

        top     = min(left_top[1], right_top[1])
        bottom  = max(left_bottom[1], right_bottom[1])
        left    = min(left_top[0], left_bottom[0])
        right   = max(right_top[0], right_bottom[0])

        return int(top), int(bottom), int(left), int(right)

    def __apply_random_margins(self, margin, rows, cols, top, bottom, right, left):
        top     += random.randint(-1 * margin, margin)
        bottom  += random.randint(-1 * margin, margin)
        right   += random.randint(-1 * margin, margin)
        left    += random.randint(-1 * margin, margin)

        top = max(0, top)
        left = max(0, left)
        bottom = min(rows, bottom)
        right = min(cols, right)
        top = min(top, bottom)
        left = min(left, right)

        return top, bottom, right, left

    def __apply_contrast(self, staff):
        clahe = cv2.createCLAHE(self.__params['clipLimit'])
        lab = cv2.cvtColor(staff, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def __apply_erosion_dilation(self, staff):
        n = random.randint(-1 * self.__params['kernel'], self.__params['kernel'])
        kernel = np.ones((abs(n), abs(n)), np.uint8)

        if(n < 0):
            return cv2.erode(staff, kernel, iterations=1)

        return cv2.dilate(staff, kernel, iterations=1)
        

    def __generate_staff_json(self, ):
        data = {}

        return data

    def modify_staffs(self):
        lines = open(self.__params['lst_rute'], 'r').read().splitlines()

        vocabulary = set()    
        x = []
        y = []

        for line in lines:
            imag_path, json_path = line.split('\t')
            img = cv2.imread(imag_path)[:,:,::-1]

            print('Loading', json_path)
            if img is not None:
                with open(json_path) as img_json:
                    data = json.load(img_json)

                    (rows, cols) = img.shape[:2]
                    img = np.pad(img, ((int(cols * self.__params['pad']),), (int(rows * self.__params['pad']),), (0,)), 'mean')
                    (new_rows, new_cols) = img.shape[:2]
                    center = (int(new_cols/2), int(new_rows/2))

                    for page in data['pages']:
                        if "regions" in page:
                            for region in page['regions']:
                                if region['type'] == 'staff' and "symbols" in region:

                                    symbol_sequence = [s["agnostic_symbol_type"] + ":" + s["position_in_straff"] for s in region["symbols"]]
                                    vocabulary.update(symbol_sequence)

                                    for _ in range(0, self.__params['iterations']):
                                        image = img
                                        y.append(symbol_sequence)

                                        staff_top, staff_left, staff_bottom, staff_right = region["bounding_box"]["fromY"], region["bounding_box"]["fromX"], region["bounding_box"]["toY"], region["bounding_box"]["toX"]

                                        staff_top     += int(cols * self.__params['pad'])
                                        staff_bottom  += int(cols * self.__params['pad'])
                                        staff_right   += int(rows * self.__params['pad'])
                                        staff_left    += int(rows * self.__params['pad'])

                                        if self.__params.get("rotation_rank"):
                                            angle = random.randint(-1 * self.__params['rotation_rank'], self.__params['rotation_rank'])
                                        else:
                                            angle = 0

                                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                        image = cv2.warpAffine(image, M, (new_cols, new_rows))

                                        M = cv2.getRotationMatrix2D(center, angle * -1, 1.0)
                                        staff_top, staff_bottom, staff_left, staff_right = self.__rotate_points(M, center, staff_top, staff_bottom, staff_left, staff_right)

                                        if self.__params.get("random_margin"):
                                            staff_top, staff_bottom, staff_right, staff_left = self.__apply_random_margins(self.__params['random_margin'], new_rows, new_cols, staff_top, staff_bottom, staff_right, staff_left)

                                        staff = image[staff_top:staff_bottom, staff_left:staff_right]


                                        if self.__params.get("contrast") == True:
                                            staff = self.__apply_contrast(staff)

                                        if self.__params.get("erosion_dilation") == True:
                                            staff = self.__apply_erosion_dilation(staff)

                                        x.append(staff)

        return x, y, vocabulary