from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from custem_image_generater import CustomDataGenerator

def create_model(input_shape, dataset_path = 'dataset_new', dataset_test_path='test', batch_size = 32):
    train_datagen = CustomDataGenerator()
    train_generator = train_datagen.flow_from_directory_with_annotations(dataset_path, input_shape, batch_size=batch_size)

    validation_generator = train_datagen.flow_from_directory_with_annotations(dataset_test_path, input_shape, batch_size=batch_size)

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(8)  # 8 чисел для координат полигона
    ])


    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss',  # Следим за изменением loss на валидации
                               patience=3,         # Количество эпох без улучшений после которых тренировка будет остановлена
                               verbose=1,           # Будет печатать сообщение при остановке
                               mode='min',          # Остановка когда значение monitor перестанет уменьшаться
                               restore_best_weights=True)  # Восстановление весов с лучшего шага

    # Обучение модели с тонкой настройкой
    history = model.fit(train_generator, 
                        epochs=5, 
                        validation_data=validation_generator, 
                        callbacks=[early_stopping], 
                        steps_per_epoch=batch_size, 
                        validation_steps=batch_size)

    # Сохранение модели
    model.save('polygon_document_model-2.2.keras')

    # Оценка модели
    test_loss, test_acc = model.evaluate(validation_generator, steps=batch_size)
    print('Точность на тестовом наборе данных:', test_acc)

