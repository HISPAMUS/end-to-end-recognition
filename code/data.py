import cv2
import json
from math import ceil
import numpy as np
import random
from sklearn.utils import shuffle
import tensorflow as tf
from threading import Lock


# ===================================================


class ImageCache:
    'Keeps a cache of images using their path as key'

    def __init__(self):
        self.__lock = Lock()
        self.__images = {}

    def read_image(self, path):
        'Retrieves an image from the cache'
        self.__lock.acquire()
        image = []
        if path in self.__images:
            image = self.__images[path]
        else:
            image = cv2.imread(path, True)
            self.__images[path] = image
        self.__lock.release()
        return image


# ===================================================


class LanguageIndex():
    '''
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb
    This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
    (e.g., 5 -> "dad") for each language,
    '''

    def __init__(self, lang):
        self.__lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for sequence in self.__lang:
            self.vocab.update(sequence)

        self.vocab = sorted(self.vocab)

        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


# ===================================================


class LstReader:
    def __init__(self, lst_path, sequence_delimiter=False):
        self.images = []
        self.regions = []
        self.symbols = []
        self.positions = []
        self.joint = []
        self.__load_lst(lst_path, sequence_delimiter)

    def __load_lst(self, lst_path, sequence_delimiter=False):
        lines = open(lst_path, 'r').read().splitlines()
        for line in lines:
            page_path, json_path = line.split('\t')
            with open(json_path) as json_file:
                region_count = 0
                data = json.load(json_file)
                for page in data['pages']:
                    if 'regions' in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and 'symbols' in region:
                                self.images.append(page_path)

                                symbols = region['symbols']
                                symbols.sort(
                                    key=lambda symbol: symbol['bounding_box']['fromX'])
                                if sequence_delimiter:
                                    self.symbols.append(
                                        ['<s>'] + [s['agnostic_symbol_type'] for s in symbols] + ['<e>'])
                                    self.positions.append(
                                        ['<s>'] + [s["position_in_straff"] for s in symbols] + ['<e>'])
                                    self.joint.append(['<s>'] + ['{}:{}'.format(
                                        s['agnostic_symbol_type'], s["position_in_straff"]) for s in symbols] + ['<e>'])
                                else:
                                    self.symbols.append(
                                        [s['agnostic_symbol_type'] for s in symbols])
                                    self.positions.append(
                                        [s["position_in_straff"] for s in symbols])
                                    self.joint.append(['{}:{}'.format(
                                        s['agnostic_symbol_type'], s["position_in_straff"]) for s in symbols])

                                top, left, bottom, right = region['bounding_box']['fromY'], region['bounding_box']['fromX'], region['bounding_box']['toY'], region['bounding_box']['toX']
                                region_id = region['id']
                                self.regions.append(
                                    [top, bottom, left, right, region_id])
                                region_count += 1
                #print('{}: {} regions'.format(json_path, region_count))
                # if region_count == 0:
                #    print('No regions found in {}'.format(json_path))

        self.symbol_lang = LanguageIndex(self.symbols)
        for i, seq in enumerate(self.symbols):
            self.symbols[i] = [self.symbol_lang.word2idx[word] for word in seq]

        self.position_lang = LanguageIndex(self.positions)
        for i, seq in enumerate(self.positions):
            self.positions[i] = [self.position_lang.word2idx[word]
                                 for word in seq]

        self.joint_lang = LanguageIndex(self.joint)
        for i, seq in enumerate(self.joint):
            self.joint[i] = [self.joint_lang.word2idx[word] for word in seq]


# ===================================================


class StaffsModificator:
    __params = dict()
    __params['pad'] = 0.1

    # Contrast
    __params['clipLimit'] = 1.0

    # Erosion and Dilation
    __params['kernel'] = 4

    def __init__(self, **options):
        self.__params['rotation_rank'] = options['rotation'] if options.get(
            "rotation") else 0
        self.__params['random_margin'] = options['margin'] if options.get(
            "margin") else 0
        self.__params['erosion_dilation'] = options['erosion_dilation'] if options.get(
            "erosion_dilation") else False
        self.__params['contrast'] = options['contrast'] if options.get(
            "contrast") else False
        self.__params['iterations'] = options['iterations'] if options.get(
            "iterations") else 1

    def __getRegion(self, region, rows, cols):
        staff_top, staff_left, staff_bottom, staff_right = region["bounding_box"]["fromY"], region[
            "bounding_box"]["fromX"], region["bounding_box"]["toY"], region["bounding_box"]["toX"]

        staff_top += int(cols * self.__params['pad'])
        staff_bottom += int(cols * self.__params['pad'])
        staff_right += int(rows * self.__params['pad'])
        staff_left += int(rows * self.__params['pad'])

        return staff_top, staff_left, staff_bottom, staff_right

    def __rotate_point(self, M, center, point):
        point[0] -= center[0]
        point[1] -= center[1]

        point = np.dot(point, M)

        point[0] += center[0]
        point[1] += center[1]

        return point

    def __rotate_points(self, M, center, top, bottom, left, right):
        left_top = self.__rotate_point(M, center, [left, top])
        right_top = self.__rotate_point(M, center, [right, top])
        left_bottom = self.__rotate_point(M, center, [left, bottom])
        right_bottom = self.__rotate_point(M, center, [right, bottom])

        top = min(left_top[1], right_top[1])
        bottom = max(left_bottom[1], right_bottom[1])
        left = min(left_top[0], left_bottom[0])
        right = max(right_top[0], right_bottom[0])

        return int(top), int(bottom), int(left), int(right)

    def __apply_random_margins(self, margin, rows, cols, top, bottom, right, left):
        top += random.randint(-1 * margin, margin)
        bottom += random.randint(-1 * margin, margin)
        right += random.randint(-1 * margin, margin)
        left += random.randint(-1 * margin, margin)

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
        n = random.randint(-1 *
                           self.__params['kernel'], self.__params['kernel'])
        kernel = np.ones((abs(n), abs(n)), np.uint8)

        if(n < 0):
            return cv2.erode(staff, kernel, iterations=1)

        return cv2.dilate(staff, kernel, iterations=1)

    def apply(self, img, top, bottom, left, right):
        # print("Modificando...")
        (rows, cols) = img.shape[:2]
        img = np.pad(img, ((int(
            cols * self.__params['pad']),), (int(rows * self.__params['pad']),), (0,)), 'mean')
        (new_rows, new_cols) = img.shape[:2]
        center = (int(new_cols/2), int(new_rows/2))

        top += int(cols * self.__params['pad'])
        bottom += int(cols * self.__params['pad'])
        right += int(rows * self.__params['pad'])
        left += int(rows * self.__params['pad'])

        if self.__params.get("rotation_rank"):
            angle = random.randint(
                -1 * self.__params['rotation_rank'], self.__params['rotation_rank'])
        else:
            angle = 0

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(img, M, (new_cols, new_rows))

        M = cv2.getRotationMatrix2D(center, angle * -1, 1.0)
        top, bottom, left, right = self.__rotate_points(
            M, center, top, bottom, left, right)

        if self.__params.get("random_margin"):
            top, bottom, right, left = self.__apply_random_margins(
                self.__params['random_margin'], new_rows, new_cols, top, bottom, right, left)

        staff = image[top:bottom, left:right]

        if self.__params.get("contrast") == True:
            staff = self.__apply_contrast(staff)

        if self.__params.get("erosion_dilation") == True:
            staff = self.__apply_erosion_dilation(staff)

        return staff


# ===================================================


class DataReader:
    def __init__(self,
                 lst_path,
                 image_height=64,
                 sequence_delimiter=False,
                 channels=1,
                 test_split=0.1,
                 validation_split=0.1,
                 batch_size=16,
                 image_transformations=4,
                 seed=None,
                 train_limit=None,
                 parallel=tf.data.experimental.AUTOTUNE):

        self.__lst = LstReader(lst_path, sequence_delimiter)
        self.__augmenter = StaffsModificator(
            rotation=3, margin=10, erosion_dilation=True, contrast=False)
        self.__TRANSFORMATIONS = image_transformations
        self.__cache = ImageCache()
        self.__IMAGE_HEIGHT = image_height
        self.__CHANNELS = channels
        self.__PARALLEL = parallel
        self.__BATCH_SIZE = batch_size
        self.DATA_SIZE = len(self.__lst.regions)
        self.TEST_SPLIT = np.uint32(self.DATA_SIZE * test_split)
        self.VAL_SPLIT = np.uint32(self.DATA_SIZE * test_split)
        self.TRAIN_SPLIT = np.uint32((self.DATA_SIZE - self.TEST_SPLIT - self.VAL_SPLIT) * image_transformations)
        if train_limit is not None:
            self.TRAIN_SPLIT = np.uint32(train_limit * image_transformations)

        images, regions, symbols, positions, joint = shuffle(self.__lst.images,
                                                             self.__lst.regions,
                                                             self.__lst.symbols,
                                                             self.__lst.positions,
                                                             self.__lst.joint,
                                                             random_state=seed)

        test_idx = self.TEST_SPLIT
        val_idx = test_idx + self.VAL_SPLIT
        train_idx = self.DATA_SIZE
        if train_limit is not None:
            train_idx = val_idx + train_limit

        self.__image_test_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in images[:test_idx]], tf.string)
        # Workaround for creating a dataset with sequences of different length
        self.__region_test_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in regions[:test_idx]], tf.int32)
        self.__symbol_test_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in symbols[:test_idx]], tf.int32)
        self.__position_test_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in positions[:test_idx]], tf.int32)
        self.__joint_test_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in joint[:test_idx]], tf.int32)

        self.__image_val_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in images[test_idx:val_idx]], tf.string)
        # Workaround for creating a dataset with sequences of different length
        self.__region_val_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in regions[test_idx:val_idx]], tf.int32)
        self.__symbol_val_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in symbols[test_idx:val_idx]], tf.int32)
        self.__position_val_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in positions[test_idx:val_idx]], tf.int32)
        self.__joint_val_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in joint[test_idx:val_idx]], tf.int32)

        images_train, regions_train, symbols_train, positions_train, joint_train = [], [], [], [], []
        for _ in range(self.__TRANSFORMATIONS):
            images_train = images_train + images[val_idx:train_idx]
            regions_train = regions_train + regions[val_idx:train_idx]
            symbols_train = symbols_train + symbols[val_idx:train_idx]
            positions_train = positions_train + positions[val_idx:train_idx]
            joint_train = joint_train + joint[val_idx:train_idx]

        self.__image_train_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in images_train], tf.string)
        # Workaround for creating a dataset with sequences of different length
        self.__region_train_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in regions_train], tf.int32)
        self.__symbol_train_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in symbols_train], tf.int32)
        self.__position_train_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in positions_train], tf.int32)
        self.__joint_train_ds = tf.data.Dataset.from_generator(
            lambda: [(yield _) for _ in joint_train], tf.int32)

    def get_data(self):
        shapes = (tf.TensorShape([self.__IMAGE_HEIGHT, None, 1]), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape(
            [None]), tf.TensorShape([None]), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()))
        test_ds = tf.data.Dataset.zip((self.__image_test_ds, self.__region_test_ds, self.__symbol_test_ds, self.__position_test_ds, self.__joint_test_ds)) \
            .map(self.__map_load_and_preprocess_regions, num_parallel_calls=self.__PARALLEL) \
            .padded_batch(self.__BATCH_SIZE, padded_shapes=shapes, padding_values=(0., 0, -1, -1, -1, 0, '', 0)) \
            .prefetch(1)
        val_ds = tf.data.Dataset.zip((self.__image_val_ds, self.__region_val_ds, self.__symbol_val_ds, self.__position_val_ds, self.__joint_val_ds)) \
            .map(self.__map_load_and_preprocess_regions, num_parallel_calls=self.__PARALLEL) \
            .padded_batch(self.__BATCH_SIZE, padded_shapes=shapes, padding_values=(0., 0, -1, -1, -1, 0, '', 0)) \
            .prefetch(1)
        train_ds = tf.data.Dataset.zip((self.__image_train_ds, self.__region_train_ds, self.__symbol_train_ds, self.__position_train_ds, self.__joint_train_ds)) \
            .map(self.__map_load_and_preprocess_modificated_regions, num_parallel_calls=self.__PARALLEL) \
            .padded_batch(self.__BATCH_SIZE, padded_shapes=shapes, padding_values=(0., 0, -1, -1, -1, 0, '', 0)) \
            .prefetch(1)
        return train_ds, val_ds, test_ds

    def get_dictionaries(self):
        return self.__lst.symbol_lang, self.__lst.position_lang, self.__lst.joint_lang

    def __load_and_preprocess_regions(self, path, region, symbol, position, joint):
        #print('Loading region {}[{}]'.format(path, region))
        page_image = self.__cache.read_image(path.decode())
        top, bottom, left, right, region_id = region

        img = page_image[top:bottom, left:right]
        if self.__CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Pre-process
        img = np.float32((255. - img) / 255.)
        height = self.__IMAGE_HEIGHT
        width = int(float(height * img.shape[1]) / img.shape[0])
        img = cv2.resize(img, (width, height))
        if self.__CHANNELS == 1:
            img = img[:, :, np.newaxis]

        # (image, image_width, symbol_sequence, symbol_sequence_length, page_path, region_id)
        # page_path & region_id are included for debugging purposes
        return img, np.int32(width), symbol, position, joint, np.int32(len(symbol)), path, region_id

    def __map_load_and_preprocess_regions(self, image, region, symbol, position, joint):
        return tf.py_func(self.__load_and_preprocess_regions, [image, region, symbol, position, joint], [tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.int32])

    def __load_and_preprocess_modificated_regions(self, path, region, symbol, position, joint):
        #print('Loading region {}[{}]'.format(path, region))
        page_image = self.__cache.read_image(path.decode())
        top, bottom, left, right, region_id = region

        img = self.__augmenter.apply(page_image, top, bottom, left, right)

        if self.__CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Pre-process
        img = np.float32((255. - img) / 255.)
        height = self.__IMAGE_HEIGHT
        width = int(float(height * img.shape[1]) / img.shape[0])
        img = cv2.resize(img, (width, height))
        if self.__CHANNELS == 1:
            img = img[:, :, np.newaxis]

        # (image, image_width, symbol_sequence, symbol_sequence_length, page_path, region_id)
        # page_path & region_id are included for debugging purposes
        return img, np.int32(width), symbol, position, joint, np.int32(len(symbol)), path, region_id

    def __map_load_and_preprocess_modificated_regions(self, image, region, symbol, position, joint):
        if self.__TRANSFORMATIONS > 1:
            return tf.py_func(self.__load_and_preprocess_modificated_regions, [image, region, symbol, position, joint], [tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.int32])
        else:
            return tf.py_func(self.__load_and_preprocess_regions, [image, region, symbol, position, joint], [tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.int32])


# ===================================================


def prepare_data(batch, vocabulary_sizes, params):
    X, XL, Y_symbol, Y_position, Y_joint, YL, _, _ = batch
    XL = [length // params['width_reduction'] for length in XL]
    Y_symbol = [y[:YL[idx]] for idx, y in enumerate(Y_symbol)]
    Y_position = [y[:YL[idx]] for idx, y in enumerate(Y_position)]
    Y_joint = [y[:YL[idx]] for idx, y in enumerate(Y_joint)]

    # Deal with empty staff sections
    for idx, _ in enumerate(X):
        if YL[idx] == 0:
            Y_symbol[idx] = [vocabulary_sizes[0]]  # Blank CTC
            Y_position[idx] = [vocabulary_sizes[1]]  # Blank CTC
            Y_joint[idx] = [vocabulary_sizes[2]]  # Blank CTC
            YL[idx] = 1
    
    return X, XL, Y_symbol, Y_position, Y_joint, YL


# ===================================================
