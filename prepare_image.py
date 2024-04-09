import cv2
import math
import numpy as np

def contours_of_document(image, show=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение алгоритма Canny для обнаружения границ
    edges = cv2.Canny(gray, 1, 50, apertureSize=3)

    if show:   
        cv2.imshow('bw', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edges

def preprocess_image(image, show=False):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 1, 50)

    #Нахождение контуров на изображении
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Отрисовка контуров на изображении
    image = image.copy()
    cv2.drawContours(image, contours, -1, (0, 0, 0), 2)

    if show:   
        cv2.imshow('edged', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def resize_img_points(image, points, new_width, new_height):
    resized_image = cv2.resize(image, (new_width, new_height)) 
    new_points = []
    points = np.array(points)
    points = points.reshape((-1, 2))
    for point in points:
        new_x = int(point[0] * (new_width / image.shape[1])) / new_width
        new_y = int(point[1] * (new_height / image.shape[0])) / new_height
        new_points.extend([new_x, new_y])
    new_points = np.array(new_points)
    return resized_image, new_points

def back_coordinates(points, width, height):
    new_points = []
    points = np.array(points)
    points = points.reshape((-1, 2))
    for point in points:
        new_x = point[0] * width
        new_y = point[1] * height
        new_points.extend([new_x, new_y])
    new_points = np.array(new_points)
    return new_points

def rotate_polygon(points, angle_degrees, center_x, center_y):
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    rotated_points = []
    for x, y in points:
        # Смещение к началу координат
        x -= center_x
        y -= center_y
        
        # Поворот
        new_x = x * cos_angle - y * sin_angle
        new_y = x * sin_angle + y * cos_angle
        
        new_x += center_x 
        new_y += center_y
       
        new_x = round(new_x, 2)
        new_y = round(new_y, 2)
        rotated_points.append((new_x, new_y))
    
    return rotated_points

def rotate_image_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def prepare_data(image, input_shape, points):
    n = 0
    if len(input_shape) == 3:
        n = input_shape[2]

    new_width, new_height, _ = input_shape
    image = preprocess_image(image)

    if n < 1:
        image = image / 255

    image, points = resize_img_points(image, points, new_width, new_height)

    return image, points

def show_polygon(image, points):
    points = points.reshape((-1, 2))
    cv2.polylines(image, np.int32([points]), isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow('Image with Polygon', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

