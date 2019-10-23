import matplotlib.pyplot as plt
import cv2
import json

def read(file):
    x = []
    y = []

    lines = open(file, 'r').read().splitlines()

    for line in lines:
        x.append(line.split()[1])
        y.append(float(line.split("SER:")[1].split()[0]))

    return x, y

if __name__ == "__main__":
    rute = "data/hispamus_jclg.lst"

    lines = open(rute, 'r').read().splitlines()

    for line in lines:
        imag_path, json_path = line.split('\t')
        img = cv2.imread(imag_path)

        print('Loading', json_path)
        if img is not None:
            with open(json_path) as img_json:
                data = json.load(img_json)

                for page in data['pages']:
                    if "regions" in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and "symbols" in region:
                                staff_top, staff_left, staff_bottom, staff_right = region["bounding_box"]["fromY"], region["bounding_box"]["fromX"], region["bounding_box"]["toY"], region["bounding_box"]["toX"]

                                for symbol in region['symbols']:
                                    symbol_top, symbol_left, symbol_bottom, symbol_right = symbol["bounding_box"]["fromY"], symbol["bounding_box"]["fromX"], symbol["bounding_box"]["toY"], symbol["bounding_box"]["toX"]
                                    cv2.rectangle(img,(symbol_left,symbol_top),(symbol_right,symbol_bottom),(0,255,255),2)

                                cv2.rectangle(img,(staff_left,staff_top),(staff_right,staff_bottom),(200,255,0),2)

                plt.imshow(img)
                plt.show()