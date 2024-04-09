import tensorflow as tf
import cv2, json
import numpy as np
from custem_image_generater import CustomDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from prepare_image import prepare_data, show_polygon, back_coordinates, resize_img_points

def train_model(model, input_shape, dataset_path = 'dataset_new', dataset_test_path='test', batch_size = 32, save=True):
    train_datagen = CustomDataGenerator()
    train_generator = train_datagen.flow_from_directory_with_annotations(dataset_path, input_shape, batch_size=batch_size)

    validation_generator = train_datagen.flow_from_directory_with_annotations(dataset_test_path, input_shape, batch_size=batch_size)

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy',  # Следим за изменением loss на валидации
                               patience=3,         # Количество эпох без улучшений после которых тренировка будет остановлена
                               verbose=1,           # Будет печатать сообщение при остановке
                               mode='max',          # Остановка когда значение monitor перестанет уменьшаться
                               restore_best_weights=True)  # Восстановление весов с лучшего шага

    # Обучение модели с тонкой настройкой
    history = model.fit(train_generator, 
                        epochs=30, 
                        validation_data=validation_generator, 
                        callbacks=[early_stopping], 
                        steps_per_epoch=batch_size, 
                        validation_steps=batch_size)

    # Сохранение модели
    if save:
        model.save('polygon_document_model-1.1.keras')

    # Оценка модели
    test_loss, test_acc = model.evaluate(validation_generator, steps=batch_size)
    print('Точность на тестовом наборе данных:', test_acc)

def test_model(input_shape, train=False, save_model=True):
    model = tf.keras.models.load_model('polygon_document_model-1.1.keras')
    if train:
        train_model(model, input_shape, save=save_model)
    # Check its architecture
    #model.summary()

    image_m = cv2.imread("48.png")
    shape = image_m.shape
    print(shape)
    with open('48.json') as file_json:
        data = json.load(file_json)
        points = data['points']

    imag_new, points_new = prepare_data(image_m.copy(), input_shape, points)
    image = cv2.resize(image_m.copy(), input_shape[:2])

    points_new = back_coordinates(points_new, image.shape[0], image.shape[1])

    show_polygon(image.copy(), points_new)

    pred = model.predict(np.array([imag_new]))
    pred = back_coordinates(pred[0], imag_new.shape[0], imag_new.shape[1])

    _, pred = resize_img_points(image.copy(), points_new, shape[0], shape[1])

    pred = back_coordinates(pred, shape[0], shape[1])

    print(pred)

    show_polygon(image_m.copy(), pred)