from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from keras import backend as K
import numpy as np
import json

EPOCHS = 2

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# X, list of the images
X_maps = []
X_troops = []
Y = []
weights = []
cont = 0

with open('battle_info.json','r') as f:
    data = json.load(f)

for camp in data:
    cont +=1
    X_maps.append(preprocess_image(camp))
    x_pos = [x*224//768 for x in [data[camp]['troop1_x'], data[camp]['troop2_x'], data[camp]['troop3_x'], data[camp]['troop4_x']]]
    y_pos = [x*224//432 for x in [data[camp]['troop1_y'], data[camp]['troop2_y'], data[camp]['troop3_y'], data[camp]['troop4_y']]]
    X_troops.append(np.array(x_pos+y_pos))
    Y.append(data[camp]['battle_result'])
    print("%d: x=%s   y=%s  res=%d" % (cont, str(x_pos), str(x_pos), data[camp]['battle_result']))

X_maps = np.array(X_maps)
X_troops = np.array(X_troops)
Y = np.array(Y)
wins = np.count_nonzero(Y)
defeats = len(Y) - wins
cw = { # class_weight
    0: wins,
    1: defeats
}

print("%d    %d     %d" %(len(X_maps), len(X_troops), len(Y)))

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)   # lo screen del villaggio
troops_input = Input(shape=(8,), name='troops_input')   # 8 = 4*2 (4 n truppe, 2 x e y)
print(base_model.input)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
enriched_model = concatenate([x, troops_input])
# let's add a fully-connected layer
x = Dense(128, activation='relu')(enriched_model)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)


# this is the model we will train
model = Model(inputs=[base_model.input, troops_input], outputs=predictions) #TODO allenare separatamente?
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy') #TODO
model.compile(optimizer='rmsprop', loss='binary_crossentropy') #TODO

model.summary()

# train the model on the new data for a few epochs
model.fit(x=[X_maps, X_troops], y=Y, epochs=EPOCHS, class_weight=cw)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from VGG16.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True
for layer in base_model.layers:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(x=[X_maps, X_troops], y=Y, epochs=EPOCHS, class_weight=cw)
