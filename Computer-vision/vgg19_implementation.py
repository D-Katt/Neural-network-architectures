"""С применением функционального подхода производится конструирование модели,
аналогичной по архитектуре модели VGG19, из базовых слоев keras.layers.
Отличие от модели VGG16 - в количестве слоев convolution внутри блоков.
Дефолтные значения аргументов функции создают модель на 1000 классов,
принимающую на вход цветные изображения размером 224 х 224 пикселя.
Размеры "окна" для слоев convolution - 3 х 3, pooling - 2 х 2.
"""

import tensorflow as tf

# Проверка доступных GPU:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs available: {len(physical_devices)}')

# Если есть GPU:
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_block(input_data, n_neurons, n_conv=2):
    """Функция формирует блок, состоящий из нескольких слоев
    convolution и финального слоя pooling.
    Аргументы:
        input_data - входные данные из предыдущего блока или слоя
        n_neurons - количество нейронов в слоях блока
        n_conv - количество слоев convolution внутри блока
    """
    x = tf.keras.layers.Conv2D(n_neurons, 3, padding='same', activation='relu')(input_data)
    for conv_layer in range(n_conv - 1):
        x = tf.keras.layers.Conv2D(n_neurons, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    return x


def create_model(input_size=224, n_classes=1000):
    """Функция формирует и инициализирует модель."""
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))  # Входной слой
    # Блоки внутри модели:
    x = create_block(inputs, 64)  # Блок 1
    x = create_block(x, 128)  # Блок 2
    x = create_block(x, 256, n_conv=4)  # Блок 3
    x = create_block(x, 512, n_conv=4)  # Блок 4
    x = create_block(x, 512, n_conv=4)  # Блок 5
    # Преобразование данных в одномерный массив:
    x = tf.keras.layers.Flatten()(x)
    # Полносвязные слои:
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    # Финальный слой по количеству классов:
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = create_model()
model.summary()

# ------------------------- Архитектура исходной модели -------------------------------

# Model: "vgg19"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0
# _________________________________________________________________
