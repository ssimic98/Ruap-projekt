import PIL
from tensorflow import keras
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
warnings.filterwarnings('ignore')

import splitfolders
splitfolders.ratio("cats_vs_dogs", 
                  output="data", 
                  seed=42, 
                  ratio=(.7, .2, .1), 
                  group_prefix=None, 
                  move=False 
                )




input_path = []

for class_name in os.listdir("PetImages"):
    for path in os.listdir("PetImages/"+class_name):
        input_path.append(os.path.join("PetImages", class_name, path))
    
df = pd.DataFrame()
df['images'] = input_path

l = []
for image in df['images']:
    try:
        img = PIL.Image.open(image)
    except:
        l.append(image)
print(l)


trainDataGenerator = ImageDataGenerator(rescale = 1/255.0)
validDataGenerator = ImageDataGenerator(rescale = 1/255.0)

train_data = trainDataGenerator.flow_from_directory(
    directory = 'data/train',
    target_size = (224,224),
    class_mode = 'categorical',
    batch_size = 64,
    seed = 42
)
valid_data = validDataGenerator.flow_from_directory(
    directory = 'data/val',
    target_size = (224,224),
    class_mode = 'categorical',
    batch_size = 64,
    seed = 42
)




model1 = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (224, 224, 3), activation = 'relu'),
    keras.layers.MaxPool2D(pool_size = (2,2), padding = 'same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(2, activation = 'softmax')
])

model1.compile(
    loss = keras.losses.categorical_crossentropy,
    optimizer = keras.optimizers.Adam(),
    metrics = [keras.metrics.BinaryAccuracy(name = 'accuracy')]
)


history1 = model1.fit(
    train_data,
    validation_data = valid_data,
    epochs = 10
)

model1.save("model1.h5")

model2 = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (224, 224, 3), activation = 'relu'),
    keras.layers.MaxPool2D(pool_size = (2,2), padding = 'same'),
    keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(pool_size = (2,2), padding = 'same'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(2, activation = 'softmax')
])

model2.compile(
    loss = keras.losses.categorical_crossentropy,
    optimizer = keras.optimizers.Adam(),
    metrics = [keras.metrics.BinaryAccuracy(name = 'accuracy')]
)

history2 = model2.fit(
    train_data,
    validation_data = valid_data,
    epochs = 10
)

model2.save("model2.h5")























