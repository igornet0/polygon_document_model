import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os, json
from sklearn.model_selection import train_test_split
from random import shuffle


def unpack_coordinates(coordinates):
    unpacked_coordinates = []
    for cord in coordinates:
        unpacked_coordinates.extend(cord)
    result = np.array([round(x/800, 4) for x in unpacked_coordinates], np.float32)
    return result

def prepare_data(image):
    # Преобразовать изображение в черно-белое
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    resized_image = cv2.resize(gray_image, (800, 800))

    # Нормализовать значения пикселей до диапазона [0, 1]
    normalized_image = resized_image / 255.0
    return np.array([normalized_image])

def png_file_generator(folder_path):
    l = os.listdir(folder_path)
    while True:
        shuffle(l)
        for file in l:
            if file.endswith(".json"):
                with open(f'{folder_path}/{file.replace("png","json")}', 'r') as file:
                    data = json.load(file)
                    points = data['points']
                    file = data['namefile']
                    d = np.array(points).reshape(-1)
                    if len(d) > 8:
                        continue
                yield prepare_data(cv2.imread(f"{folder_path}/{file}")),unpack_coordinates(points)

# Загрузка данных и меток для обучения
data = png_file_generator('new_dataset')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(800, 800, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(8)  # 8 координат для предсказания
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
def data_test(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(f'{folder_path}/{file.replace("png","json")}', 'r') as file:
                data = json.load(file_json)["shapes"][0]
                #data = json.load(file_json)
                points = data['points']
                points = np.array(points, np.int32)
                points = points.reshape((-1, 2))
                file = data['imagePath']
                d = np.array(points).reshape(-1)
                if len(d) > 8:
                    continue
            yield prepare_data(cv2.imread(f"{folder_path}/{file}")),unpack_coordinates(points)
#model = tf.keras.models.load_model('polygon_model.keras')
# Обучение модели
model.fit(data, epochs=30, steps_per_epoch=2144, validation_data=data_test("test"))

#accuracy = model.evaluate(X_test, y_test)[1]
#print("Accuracy using RNN:", accuracy)
# Сохранение модели
model.save('polygon_model.keras')

#model = tf.keras.models.load_model('polygon_model.keras')

image = cv2.imread("48.png")
with open('48.json') as file_json:
    data = json.load(file_json)["shapes"][0]
    #data = json.load(file_json)
    points = data['points']

points = np.array(points, np.int32)
points_t = points.reshape((-1, 2)) 

points = model.predict([prepare_data(image)])[0] 
print(points)
points  = [(points[0], points[1]), [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]]
print(f"{points=}")
print(f"{points_t=}")
image = cv2.resize(image, (800, 800)) 
pr
cv2.polylines(image, np.int32(points), isClosed=True, color=(255, 0, 0), thickness=2)
cv2.imshow('Image with Polygon', image)
cv2.waitKey(0)
cv2.destroyAllWindows()