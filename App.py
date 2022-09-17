from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from tensorflow import keras

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
    
model2.load_weights('static/model2.h5')


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction = model2.predict(img_arr)

    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.config['DEBUG'] = True
    app.config['SERVER_NAME'] = "127.0.0.1:5010"         
    app.run()



