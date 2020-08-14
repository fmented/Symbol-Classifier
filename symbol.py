import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dropout, MaxPooling2D
import cv2
from os import system


csv_file    =   'symbols (5).csv'
model_name  =   '7_symbols.h5'


# PREPARE //////////////////////////////////////////////
features=[]

def b64toimage(url):
    try:  
        b_64 = url.split(',')[1]
    except AttributeError:
        url=url.read()
        b_64 = url.split(',')[1]
    enc = base64.b64decode(b_64)
    image = np.array(bytearray(enc), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    return image

df_from_csv=pd.read_csv(csv_file, sep='\t')


for index, row in df_from_csv.iterrows():
    img=b64toimage(row['image-url'])
    label=row['class']
    features.append([img, label])

img_df=pd.DataFrame(features, columns=['img', 'label'])

x = np.array(img_df.img.to_list())
y = np.array(img_df.label.to_list())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))


x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state=4)


# BUILD MODEL //////////////////////////////////////////////



num_rows = 32
num_columns = 32
num_channels = 1

x_test=x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
x_train=x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)


num_labels = yy.shape[1]

model = Sequential()
model.add(Conv2D(filters=64, strides=1 ,kernel_size=(5,5), activation='relu', input_shape=(32,32, 1)))
model.add(AveragePooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64,  strides=1,  kernel_size=(5,5), activation='relu'))
model.add(AveragePooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(units=120,activation='relu'))
model.add(Dense(units=84,activation='relu'))
model.add(Dense(num_labels, activation='softmax'))


# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())

# model.add(Dense(num_labels, activation='softmax'))

# COMPILE MODEL /////////////////////////////////////////////


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

score = model.evaluate(x_test, y_test, verbose=1)
print('pre-training accuracy {}'.format(score[1]))

# TRAIN MODEL /////////////////////////////////////////////////


num_epochs = 86
num_batch_size = 256

checkpoint = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpoint], verbose=1)

# EVALUATE ///////////////////////////////////////////////////
score = model.evaluate(x_train, y_train, verbose=0)
print('training acuracy {}'.format(score[1]))
score = model.evaluate(x_test, y_test, verbose=0)
print('testing acuracy {}'.format(score[1]))

# CONVERT TO TFJS LAYERMODEL //////////////////////////////////

system("tensorflowjs_converter --input_format keras {} js_model".format(model_name))

