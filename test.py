import cv2, json, os
import matplotlib.pyplot as plt
import numpy as np
from prepare_image import *

for file in os.listdir("test"):
    if ".png" in file:
        with open(f'test/{file.replace("png", "json")}') as file_json:
                data = json.load(file_json)["shapes"][0]
                points = data['points']

        image = cv2.imread(f"test/{file}")
        points = np.array(points, np.int32)
        points = points.reshape((-1))

        data = {"namefile": f"test/images/{file}", "points": points.tolist()}
        with open(f'test/{file.replace("png", "json")}', 'w') as file_json:
            json.dump(data, file_json)

        cv2.imwrite(f'test/{file}', image)