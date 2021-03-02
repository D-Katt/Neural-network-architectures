"""С применением функционального подхода производится конструирование моделей,
аналогичных по архитектуре моделям семейства DenseNet, из базовых слоев keras.layers.
Дефолтные значения аргументов функции создают модель на 1000 классов,
принимающую на вход цветные изображения размером 224 х 224 пикселя.
Функция принимает дополнительный аргумент в виде списка повторений конволюционных блоков
в сегментах модели, что позволяет получать на выходе различные модификации DenseNet.
Количество параметров в моделях отличается от канонических образцов, которые могут
быть получены через загрузку моделей из keras.applications, но структура графа идентична.
"""

import tensorflow as tf

# Проверка доступных GPU:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs available: {len(physical_devices)}')

# Если есть GPU:
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_transition(input_data, n_neurons: int):
    """Функция формирует переходный блок между двумя
    конволюционными блоками модели.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - количество нейронов в слоях блока
    """
    x = tf.keras.layers.BatchNormalization(axis=3)(input_data)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n_neurons, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x


def create_conv_block(input_data, n_neurons: int):
    """Функция формирует блок из двух слоев convolution
    1 x 1 и 3 x 3 с предшествующей нормализацией и активацией.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - количество нейронов в слоях блока
    """
    x = tf.keras.layers.BatchNormalization(axis=3)(input_data)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n_neurons, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=n_neurons, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    return x


def create_model(n_blocks: list, input_size=224, n_classes=1000):
    """Функция формирует и инициализирует модель. Последовательно
    вызывает функции, создающие конволюционные и переходные блоки.
    Результат каждого следующего преобразования в конволюционном блоке
    конкатенируется с результатом предыдущего блока.
    Аргументы:
        n_blocks - список, содержащий число конволюционных блоков во 2-5 сегментах модели
        input_size - размер изображения в пикселях
        n_classes - количество классов (нейронов в финальном слое)
    """
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    # Сегмент 1 (отличается от остальных)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Сегмент 2
    for _ in range(n_blocks[0]):
        x_next = create_conv_block(x, n_neurons=32)
        x = tf.keras.layers.concatenate([x, x_next], axis=3)  # Соединяем исходные данные с преобразованными
    x = create_transition(x, n_neurons=128)  # Переходный блок между конволюционными блоками
    # Сегмент 3
    for _ in range(n_blocks[1]):
        x_next = create_conv_block(x, n_neurons=32)
        x = tf.keras.layers.concatenate([x, x_next], axis=3)
    x = create_transition(x, n_neurons=256)
    # Сегмент 4
    for _ in range(n_blocks[2]):
        x_next = create_conv_block(x, n_neurons=32)
        x = tf.keras.layers.concatenate([x, x_next], axis=3)
    x = create_transition(x, n_neurons=512)
    # Сегмент 5
    for _ in range(n_blocks[3]):
        x_next = create_conv_block(x, n_neurons=32)
        x = tf.keras.layers.concatenate([x, x_next], axis=3)
    # На выходе из последнего конволюционного блока не предусмотрен переходный блок,
    # поэтому добавляем здесь нормализацию и активацию:
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Финальный слой пулинга:
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Финальный слой по количеству классов:
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# Модель, аналогичная DenseNet121
model = create_model(n_blocks=[6, 12, 24, 16])
print('DenseNet121')
model.summary()

# Модель, аналогичная DenseNet169
model = create_model(n_blocks=[6, 12, 32, 32])
print('DenseNet169')
model.summary()

# Модель, аналогичная DenseNet201
model = create_model(n_blocks=[6, 12, 48, 32])
print('DenseNet201')
model.summary()
