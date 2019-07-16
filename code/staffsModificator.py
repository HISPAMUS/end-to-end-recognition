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

    def __init__(self, conf_path = None , **options):
        self.__params['rotation_rank']    = options['rotation']         if options.get("rotation")         else 0
        self.__params['random_margin']    = options['margin']           if options.get("margin")           else 0
        self.__params['erosion_dilation'] = options['erosion_dilation'] if options.get("erosion_dilation") else False
        self.__params['contrast']         = options['contrast']         if options.get("contrast")         else False
        self.__params['fish_eye']         = options['fish_eye']         if options.get("fish_eye")         else False
        self.__params['iterations']       = options['iterations']       if options.get("iterations")       else 0

        if(conf_path != None):
            self.loadConf(conf_path)

    def loadConf(self, conf_path):
        with open(conf_path) as json_file:
            data = json.load(json_file)

            self.__params['rotation_rank']    = data['rotation_rank']    if 'rotation_rank'    in data else self.__params['rotation_rank']
            self.__params['random_margin']    = data['random_margin']    if 'random_margin'    in data else self.__params['random_margin']
            self.__params['erosion_dilation'] = data['erosion_dilation'] if 'erosion_dilation' in data else self.__params['erosion_dilation']
            self.__params['contrast']         = data['contrast']         if 'contrast'         in data else self.__params['contrast']
            self.__params['fish_eye']         = data['fish_eye']         if 'fish_eye'         in data else self.__params['fish_eye']
            self.__params['iterations']       = data['iterations']       if 'iterations'       in data else self.__params['iterations']

    def __getRegion(self, region, rows, cols):
        staff_top, staff_left, staff_bottom, staff_right = region["bounding_box"]["fromY"], region["bounding_box"]["fromX"], region["bounding_box"]["toY"], region["bounding_box"]["toX"]

        staff_top     += int(cols * self.__params['pad'])
        staff_bottom  += int(cols * self.__params['pad'])
        staff_right   += int(rows * self.__params['pad'])
        staff_left    += int(rows * self.__params['pad'])

        return staff_top, staff_left, staff_bottom, staff_right

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
        sc = (margin/100) * abs(top - bottom)

        top     += np.random.normal(scale = sc, size = 1)
        bottom  += np.random.normal(scale = sc, size = 1)
        right   += np.random.normal(scale = sc, size = 1)
        left    += np.random.normal(scale = sc, size = 1)

        # Para que no se salga de los margenes de la imagen

        top     = max(0, top)
        left    = max(0, left)
        bottom  = min(rows, bottom)
        right   = min(cols, right)
        top     = min(top, bottom)
        left    = min(left, right)

        return int(top), int(bottom), int(right), int(left)

    def __apply_contrast(self, staff):
        if(random.randint(0, 1) == 0):
            return staff

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

    def __apply_fish_eye(self, staff):
        (staff_rows, staff_cols) = staff.shape[:2]

        average = staff.mean(axis=0).mean(axis=0)

        K = np.array([[  staff_cols/2,     0.,  staff_cols/2],
                    [    0.,   staff_cols/2,       0],
                    [    0.,     0.,       1.]])

        D = np.array([0., 0., 0., 0.])

        Knew = K.copy()
        Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]

        staff = cv2.fisheye.undistortImage(staff, K, D=D, Knew=Knew)
        staff = staff[0:int(staff_rows/2), int(staff_cols * 0.18):int(staff_cols * 0.80)]

        staff = np.array(staff)
        staff[(staff == (0, 0, 0)).all(axis = -1)] = average

        return staff

    def __getStaffs2Train(self, rute, val_split):
        num_staffs = 0

        lines = open(rute, 'r').read().splitlines()

        for line in lines:
            json_path = line.split('\t')[1]

            with open(json_path) as img_json:
                data = json.load(img_json)
                for page in data['pages']:
                    if "regions" in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and "symbols" in region:
                                num_staffs += 1

        staffs2train = np.ones(num_staffs)
        idx = []

        for i in range(num_staffs):
            idx.append(i)

        random.shuffle(idx)
        idx = idx[:int(val_split * len(idx))]

        for i in idx:
            staffs2train[i] = 0

        return staffs2train

    def __getMaps(self, vocabulary):
        w2i = {}
        i2w = {}
        
        for idx, symbol in enumerate(vocabulary):
            w2i[symbol] = idx
            i2w[idx] = symbol

        return w2i, i2w

    def __normalize_data(self, x_train, y_train, x_val, y_val, w2i):
        for i in range(min(len(x_train),len(y_train))):
            for idx, symbol in enumerate(y_train[i]):
                y_train[i][idx] = w2i[symbol]

        for i in range(min(len(x_val),len(y_val))):
            for idx, symbol in enumerate(y_val[i]):
                y_val[i][idx] = w2i[symbol]

        return y_train, y_val

    def modify_staff(self, img, top, bottom, left, right):
        x = []

        (rows, cols) = img.shape[:2]
        img = np.pad(img, ((int(cols * self.__params['pad']),), (int(rows * self.__params['pad']),), (0,)), 'mean')
        (new_rows, new_cols) = img.shape[:2]
        center = (int(new_cols/2), int(new_rows/2))

        top     += int(cols * self.__params['pad'])
        bottom  += int(cols * self.__params['pad'])
        right   += int(rows * self.__params['pad'])
        left    += int(rows * self.__params['pad'])

        x.append(img[top:bottom, left:right])

        for _ in range(0, self.__params['iterations']):
            image = img
            staff_top, staff_left, staff_bottom, staff_right = top, left, bottom, right

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

            if self.__params.get("fish_eye") == True:
                staff = self.__apply_fish_eye(staff)

            x.append(staff)

        return x
    
    def get_one_staff_modification(self, img, top, bottom, left, right):
        print("Modificando...")
        (rows, cols) = img.shape[:2]
        img = np.pad(img, ((int(cols * self.__params['pad']),), (int(rows * self.__params['pad']),), (0,)), 'mean')
        (new_rows, new_cols) = img.shape[:2]
        center = (int(new_cols/2), int(new_rows/2))

        top     += int(cols * self.__params['pad'])
        bottom  += int(cols * self.__params['pad'])
        right   += int(rows * self.__params['pad'])
        left    += int(rows * self.__params['pad'])

        if self.__params.get("rotation_rank"):
                angle = random.randint(-1 * self.__params['rotation_rank'], self.__params['rotation_rank'])
        else:
            angle = 0

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(img, M, (new_cols, new_rows))

        M = cv2.getRotationMatrix2D(center, angle * -1, 1.0)
        top, bottom, left, right = self.__rotate_points(M, center, top, bottom, left, right)

        if self.__params.get("random_margin"):
            top, bottom, right, left = self.__apply_random_margins(self.__params['random_margin'], new_rows, new_cols, top, bottom, right, left)

        staff = image[top:bottom, left:right]

        if self.__params.get("contrast") == True:
            staff = self.__apply_contrast(staff)

        if self.__params.get("erosion_dilation") == True:
            staff = self.__apply_erosion_dilation(staff)

        if self.__params.get("fish_eye") == True:
                staff = self.__apply_fish_eye(staff)
        
        return staff


    def get_train_val_staffs(self, rute, val_split = 0.1):
        idx = self.__getStaffs2Train(rute, val_split)

        lines = open(rute, 'r').read().splitlines()

        vocabulary = set()
        x_train, y_train, x_val, y_val = [], [], [], []
        num_staff = 0

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

                                    o_staff_top, o_staff_left, o_staff_bottom, o_staff_right = self.__getRegion(region, rows, cols)


                                    if(idx[num_staff] == 0):
                                        x_val.append(img[o_staff_top:o_staff_bottom, o_staff_left:o_staff_right])
                                        y_val.append(symbol_sequence.copy())

                                    else:
                                        # Se guarda la copia original
                                        x_train.append(img[o_staff_top:o_staff_bottom, o_staff_left:o_staff_right])
                                        y_train.append(symbol_sequence.copy())

                                        for _ in range(0, self.__params['iterations']):
                                            image = img
                                            staff_top, staff_left, staff_bottom, staff_right = o_staff_top, o_staff_left, o_staff_bottom, o_staff_right
                                            y_train.append(symbol_sequence.copy())

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

                                            x_train.append(staff)

                                    num_staff += 1

        w2i, i2w = self.__getMaps(vocabulary)
        y_train, y_val = self.__normalize_data(x_train, y_train, x_val, y_val, w2i)

        return x_train, y_train, x_val, y_val, w2i, i2w