"""
MNIST opdracht D: "Localisatie"      (by Marius Versteegen, 2021)
Bij deze opdracht passen we het in opdracht C gevonden model toe om
de posities van handgeschreven cijfers in een plaatje te localiseren.
"""
import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import random

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# print(x_test[0])

# print("show image\n")
# plt.figure()
# plt.imshow(x_test[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Now, for a test on the end, let's create a few large images and see if we can detect the digits
# (we do that here because the dimensions of the images are not extended yet)
largeTestImage = np.zeros(shape=(1024, 1024), dtype=np.float32)
# let's fill it with a random selection of test-samples:
nofDigits = 40
for i in range(nofDigits):
    # insert an image
    digitWidth = x_test[0].shape[0]
    digitHeight = x_test[0].shape[1]
    xoffset = random.randint(0, largeTestImage.shape[0] - digitWidth)
    yoffset = random.randint(0, largeTestImage.shape[1] - digitWidth)
    largeTestImage[xoffset:xoffset + digitWidth, yoffset:yoffset + digitHeight] = x_test[i][0:, 0:]

largeTestImages = np.array([largeTestImage])  # for now, I only test with one large test image.

# print("show largeTestImage\n")
# plt.figure()
# plt.imshow(largeTestImages[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
matplotlib.image.imsave('largeTestImage.png', largeTestImages[0])

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
largeTestImages = np.expand_dims(largeTestImages,
                                 -1)  # one channel per color component in lowest level. In this case, thats one.

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# change shape 60000,10 into 60000,1,1,10  which is 10 layers of 1x1 pix images, which I use for categorical classification.
y_train = np.expand_dims(np.expand_dims(y_train, -2), -2)
y_test = np.expand_dims(np.expand_dims(y_test, -2), -2)

"""
Opdracht D1: 

Kopieer in het onderstaande model het model dat resulteerde
in opdracht C. Gebruik evt de weights-file uit opdracht C, om trainingstijd in te korten.

Bestudeer de output files: largeTestImage.png en de Featuremaps pngs.
* Leg uit hoe elk van die plaatjes tot stand komt en verklaar aan de hand van een paar
  voorbeelden wat er te zien is in die plaatjes.
* Als het goed is zie je dat de pooling layers in grote lijnen bepalen hoe groot de 
  uiteindelijke output bitmaps zijn. Hoe werkt dat?

De convolutie lagen zorgen ervoor dat bepaalde features naar boven komen bij een afbeelding.
Door het trainen kan elk feature worden gekoppeld aan een getal bij de output. Hierdoor zal het eerste filter
zich richten op features voor het cijfer 0, 2de filter cijfer 1 etc. Bij A_1 is ook te zien dat de nullen
sterk worden afgebeeld vergeleken met andere getallen, de 6 ook deels aangezien er een cirkel in zit.
Het netwerk is dus zeker van het feit dat er bij de features dan het goede getal staan. 

De pooling zorgt voor het verkleinen van de shape. Als je een filter van 2x2 hebt halveert de pooling layer het
aantal inputs. 
"""

"""
Opdracht D2:

* Met welke stappen zou je uit de featuremap plaatjes lijsten kunnen 
genereren met de posities van de cijfers?
Door elk cijfer wat juist is een bounding box omheen zetten en deze coordinaten vervolgens in een lijst zetten.
"""

"""
Opdracht D3: 

* Hoeveel elementaire operaties zijn er in totaal nodig voor de convoluties voor 
  onze largeTestImage? (Tip: kijk terug hoe je dat bij opdracht C berekende voor 
  de kleinere input image)
* Hoe snel zou een grafische kaart met 1 Teraflops dat theoretisch kunnen uitrekenen?
* Waarom zal het in de praktijk wat langer duren?

Layer 1
Aantal * operaties: 1020 * ( 5 * 5 ) = 25500 * 20 = 510000
Aantal + operaties: 1020 * ( 5 * 5 - 1) = 24480 * 20 = 489600

Layer 2
Aantal * operaties: 508 * ( 3 * 3 ) = 4572 * 20 = 91440
Aantal + operaties: 508 * ( 3 * 3 - 1) = 4046 * 20 = 81280

Layer 3
Aantal * operaties: 250 * ( 5 * 5 ) = 6250 * 10 = 62500
Aantal + operaties: 250 * ( 5 * 5 - 1) = 6000 * 10 = 60000

Totaal = 1294820
1 teraflop = one trillion floating-point operations per second
1 float == 4 bytes
1294820 * 4 = 5179280 bytes
5179280 / 1.000.000.000.000 = 0,0000051792 seconden = 5.2 microseconden
In werkelijkheid zal dit langer duren door vertraging van de verschillende abstractie lagen in het OS 
maar ook het ophalen en opslaan van de data in het cache geheugen. Daarnaast zijn floating point operations ook:
- en /. Deze worden hier niet in meegenomen maar gebeuren wel.
"""


def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),  # 28x28
            layers.Conv2D(20, kernel_size=(5, 5), padding="valid"),  # 24x24
            layers.MaxPooling2D(pool_size=(2, 2)),  # 12x12
            layers.Conv2D(20, kernel_size=(3, 3), padding="valid"),  # 10x10
            layers.MaxPooling2D(pool_size=(2, 2)),  # 5x5
            layers.Conv2D(num_classes, kernel_size=(5, 5), padding="valid")  # 1x1
        ]
    )
    return model


model = buildMyModel(x_train[0].shape)
model.summary()

"""
## Train the model
"""

batch_size = 10  # Larger means faster training, but requires more system memory.
epochs = 100  # for now

bInitialiseWeightsFromFile = False  # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat ook de afwezigheid van een cijfer een valide mogelijkheid is.
optimizer = keras.optimizers.Adam(lr=learningrate)  # lr=0.01 is king
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("x_train.shape")
print(x_train.shape)

print("y_train.shape")
print(y_train.shape)

if (bInitialiseWeightsFromFile):
    model.load_weights(
        "Spoiler_C_weights.h5");  # let's continue from where we ended the previous time. That should be possible if we did not change the model.

# # NB: validation split van 0.2 ipv 0.1 gaf ook een boost: van 99.49 naar 99.66
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", ViewTools_NN.getColoredText(255, 255, 0, score[1]))

model.summary()

model.save_weights('myWeights.h5')

prediction = model.predict(x_test)
print(prediction[0])
print(y_test[0])

# summarize feature map shapes
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)

print(x_test.shape)

# study the meaning of the filtered outputs by comparing them for
# multiple samples
nLastLayer = len(model.layers) - 1
nLayer = nLastLayer
print("lastLayer:", nLastLayer)

baseFilenameForSave = None
x_test_flat = None
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
# ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)

lstWeights = []
for nLanger in range(len(model.layers)):
    lstWeights.append(model.layers[nLayer].get_weights())

# rebuild model, but with larger input:
model2 = buildMyModel(largeTestImages[0].shape)
model2.summary()

for nLayer in range(len(model2.layers)):
    model2.layers[nLayer].set_weights(model.layers[nLayer].get_weights())

prediction2 = model2.predict(largeTestImages)
baseFilenameForSave = 'FeaturemapOfLargeTestImage_A_'
ViewTools_NN.printFeatureMapsForLayer(nLastLayer, model2, None, largeTestImages, 0, baseFilenameForSave)
