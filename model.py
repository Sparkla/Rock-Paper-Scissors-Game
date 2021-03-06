import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True )
training_set = train_datagen.flow_from_directory(
            'C:\\Users\\Shruti\\Desktop\\Work\\Stone Paper Scissors\\Rock-Paper-Scissors\\train',
            target_size=(64,64),
            batch_size=32,
            class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255 )
test_set = train_datagen.flow_from_directory(
            'C:\\Users\\Shruti\\Desktop\\Work\\Stone Paper Scissors\\Rock-Paper-Scissors\\test',
            target_size=(64,64),
            batch_size=32,
            class_mode='categorical')


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=3,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x = training_set, validation_data = test_set, epochs= 15)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")