import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from scipy.ndimage.interpolation import shift

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

model.save('mnist_model.h5')

def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift_image(img, sx, sy):
    shifted_img = shift(img, [sy, sx], mode='constant')
    return shifted_img

def imageprepare(image):
    img = image.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = (img_array > 0.5) * 1.0
    shiftx, shifty = get_best_shift(img_array)
    img_array = shift_image(img_array, shiftx, shifty)
    return img_array

window = tk.Tk()
window.title("Распознавание рукописных цифр")

canvas_width = 200
canvas_height = 200
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

image1 = Image.new("RGB", (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image1)

def paint(event):
    brush_size = 15
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill='black')

canvas.bind("<B1-Motion>", paint)

def clear_canvas():
    canvas.delete('all')
    draw.rectangle([0, 0, canvas_width, canvas_height], fill='white')

def predict_digit():
    img_array = imageprepare(image1)
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    print('Распознанная цифра:', np.argmax(prediction))

btn_clear = tk.Button(window, text="Очистить", command=clear_canvas)
btn_clear.pack()

btn_predict = tk.Button(window, text="Предсказать", command=predict_digit)
btn_predict.pack()

window.mainloop()

