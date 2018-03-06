from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K
from keras.utils import plot_model
import pydot
import numpy as np
import json
from keras.optimizers import SGD

EPOCHS = 2

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 72))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# X, list of the images
X_maps = []
X_troops = []
Y = []
Y_res = []
weights = []
cont = 0

datagen = image.ImageDataGenerator(
featurewise_center=False,
featurewise_std_normalization=False,
rotation_range=10,
width_shift_range=0.05,
height_shift_range=0.05,
horizontal_flip=False)

with open('battle_info.json','r') as f:
    data = json.load(f)

for camp in data:
    cont +=1
    X_maps.append(preprocess_image(camp))

    x_pos = [x*128//768 for x in [data[camp]['troop1_x'], data[camp]['troop2_x'], data[camp]['troop3_x'], data[camp]['troop4_x']]]
    y_pos = [x*72//432 for x in [data[camp]['troop1_y'], data[camp]['troop2_y'], data[camp]['troop3_y'], data[camp]['troop4_y']]]
    X_troops.append(np.array(x_pos+y_pos))
    app = np.zeros(9)
    app[data[camp]['stronghold_level']-1] = 1
    Y.append(app)
    Y_res.append(data[camp]['battle_result'])
    # print("%d: x=%s   y=%s  res=%d" % (cont, str(x_pos), str(y_pos), data[camp]['battle_result']))

X_maps = np.array(X_maps)
X_troops = np.array(X_troops)
Y = np.array(Y)
Y_res = np.array(Y_res)
print (Y_res)
wins = np.count_nonzero(Y_res)
defeats = len(Y) - wins
cw = { # class_weight
    0: wins,
    1: defeats
}
print("wins: {}, defeats: {}".format(wins, defeats))

# print("%d    %d     %d" %(len(X_maps), len(X_troops), len(Y)))

# create the base pre-trained model
img_input = Input(shape=(128,72,3,), name="img_input")
conv1 = Conv2D(32, (3,3), name="conv1", activation="relu")(img_input)
drop = Dropout(0.2)(conv1)
conv2 = Conv2D(32, (3,3), name="conv2", activation="relu")(drop)
maxP = MaxPooling2D(pool_size=(2,2))(conv2)
conv3 = Conv2D(32, (3,3), name="conv3", activation="relu")(maxP)
drop2 = Dropout(0.2)(conv3)
conv4 = Conv2D(32, (3,3), name="conv4", activation="relu")(drop2)
maxP2 = MaxPooling2D(pool_size=(2,2))(conv4)
flat = Flatten()(maxP2)
dense1 = Dense(64, activation="relu", name="dense1")(flat)
dense_out = Dense(1, activation="sigmoid", name="dense_out")(dense1)

model = Model(inputs=img_input, outputs=dense_out)
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9, decay=0.1), loss='binary_crossentropy') # rmsprop

# plot_model(model, to_file='model.png')
# plot_model(model, to_file='model_layers.png', show_shapes=True)
# print("STAMPATO")

# model.summary()

datagen.fit(X_maps)

for e in range(40):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(X_maps, Y_res, batch_size=32):
        model.fit(x_batch, y_batch, class_weight=cw)
        batches += 1
        if batches >= len(X_maps) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break




# model.load_weights("weights_guess_sh.h5")
# model.fit(x=X_maps, y=Y, epochs=20)
# for i in range(2):
#     model.train_on_batch(x=X_maps[1:], y=Y[1:])
#     print(model.get_weights()[1])

# model.save_weights("weights_guess_sh.h5", overwrite=True)
ok = 0.
w = 0
for i in range(len(X_maps)):
    res = model.predict({'img_input': np.expand_dims(X_maps[i], axis=0)})
    # pred = np.argmax(res)
    pred = res[0][0] > 0.5
    if pred: w += 1
    real = Y_res[i]
    if pred==real:
        ok+=1
    print ("pred={}, real={}, {}".format(pred, real, pred==real))
print ("\nSUMMARY:\n\nperc: {} %  true: {}    false: {}   total: {}".format(ok/len(X_maps)*100, w, len(X_maps)-w, len(X_maps)))
