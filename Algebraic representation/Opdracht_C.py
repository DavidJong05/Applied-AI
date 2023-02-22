"""
MNIST opdracht C: "Only Conv"      (by Marius Versteegen, 2021)

Bij deze opdracht gebruik je geen dense layer meer.
De output is nu niet meer een vector van 10, maar een
plaatje van 1 pixel groot en 10 lagen diep.

Deze opdracht bestaat uit vier delen: C1 tm C4 (zie verderop)
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

#print(x_test[0])

#print("show image\n")
#plt.figure()
#plt.imshow(x_test[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Conv layers expect images.
# Make sure the images have shape (28, 28, 1).
# (for RGB images, the last parameter would have been 3)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# change shape 60000,10 into 60000,1,1,10  which is 10 layers of 1x1 pix images, which I use for categorical classification.
y_train = np.expand_dims(np.expand_dims(y_train,-2),-2)
y_test = np.expand_dims(np.expand_dims(y_test,-2),-2)

"""
Opdracht C1: 
    
Voeg ALLEEN Convolution en/of MaxPooling2D layers toe aan het onderstaande model.
(dus GEEN dense layers, ook niet voor de output layer)
Probeer een zo eenvoudig mogelijk model te vinden (dus met zo min mogelijk parameters)
dat een test accurracy oplevert van tenminste 0.98.

Voorbeelden van layers:
    layers.Conv2D(getal, kernel_size=(getal, getal))
    layers.MaxPooling2D(pool_size=(getal, getal))

Beschrijf daarbij met welke stappen je bent gekomen tot je model,
en beargumenteer elk van je stappen.

BELANGRIJK (ivm opdracht D, hierna):  
* Zorg er dit keer voor dat de output van je laatste layer bestaat uit een 1x1 image met 10 lagen.
Met andere woorden: zorg ervoor dat de output shape van de laatste layer gelijk is aan (1,1,10)
De eerste laag moet 1 worden bij het cijfer 0, de tweede bij het cijfer 1, etc.

Tip: Het zou kunnen dat je resultaat bij opdracht B al aardig kunt hergebruiken,
     als je flatten en dense door een conv2D vervangt.
     Om precies op 1x1 output uit te komen kun je puzzelen met de padding, 
     de conv kernal size en de pooling.
     
* backup eventueel na de finale succesvolle training van je model de gegenereerde weights file
  (myWeights.m5). Die kun je dan in opdracht D inladen voor snellere training.
  
  
Spoiler-mogelijkheid:
Mocht het je te veel tijd kosten (laten we zeggen meer dan een uur), dan
mag je de configuratie uit Spoiler_C.py gebruiken/kopieren.

Probeer in dat geval te beargumenteren waarom die configuratie een goede keuze is.
(n - f - 1)x(n - f - 1) = conv2d met padding
Omdat de afbeelding op deze manier goed gedownsized wordt zonder veel belangrijke waardes te verliezen
De afbeelding wordt niet te snel naar 1x1 gezet voor betere waardes aan het einde van de downsizing
Na 5x5 wordt pooling ook niet meer verstandig. 
"""

def buildMyModel(inputShape):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape), #28x28
            layers.Conv2D(20, kernel_size=(5, 5), padding="valid"), #24x24
            layers.MaxPooling2D(pool_size=(2, 2)), #12x12
            layers.Conv2D(20, kernel_size=(3, 3), padding="valid"), #10x10
            layers.MaxPooling2D(pool_size=(2, 2)), #5x5
            layers.Conv2D(num_classes, kernel_size=(5, 5), padding="valid") #1x1

        ]
    )
    return model

model = buildMyModel(x_train[0].shape)
model.summary()

"""
Opdracht C2: 
    
Kopieer bovenstaande model summary en verklaar bij 
bovenstaande model summary bij elke laag met kleine berekeningen 
de Output Shape

 conv2d_9 (Conv2D)           (None, 24, 24, 20)        520       
        inputs 1, outputs (20 * 5 * 5) =  500, biases 20 want 20 filters , 1 * 500 + 20 = 520 
        image = (28 - 5 + 1)x(28 - 5 + 1) = 24x24
        * operaties = 24 * (5 * 5) = 600 * 20 = 12000
        + operaties = 24 * (5 * 5 - 1) = 576 * 20 = 11520

 max_pooling2d_6 (MaxPooling2D)  (None, 12, 12, 20)       0                                                                      
        image /2 = 12x12

 conv2d_10 (Conv2D)          (None, 10, 10, 20)        3620      
        inputs 20, outputs (20 * 3 * 3), biases 20, 20 * 20 * 3 * 3 + 20 = 3620
        image = (12 - 3 + 1)x(12 - 3 + 1) = 10x10
        * operaties = 10 * (3 * 3) = 90 * 20 = 1800
        + operaties = 10 * (3 * 3 - 1) = 80 * 20 = 1600

 max_pooling2d_7 (MaxPooling2D)  (None, 5, 5, 20)         0                                                                      
        image /2 = 5x5

 conv2d_11 (Conv2D)          (None, 1, 1, 10)          5010 
        inputs 20, outputs (10 * 5 * 5), biases 10, 20 * 10 * 5 * 5 + 10 =  5010
        image = (5 - 5 + 1)x(5 - 5 + 1) = 1x1
        * operaties = 1 * (5 * 5) = 25 * 20 = 500
        + operaties = 1 * (5 * 5 - 1) = 24 * 20 = 480
"""

"""
Opdracht C3: 
    
Verklaar nu bij elke laag met kleine berekeningen het aantal parameters.
zie C2 , voor aantal params in conv2d: inputs * filtersize * amount_of_filter + bias_for_each_filter = params
"""

"""
Opdracht C4: 
    
Bij elke conv layer hoort een aantal elementaire operaties (+ en *).
* Geef PER CONV LAYER een berekening van het totaal aantal operaties 
  dat nodig is voor het klassificeren van 1 test-sample.
* Op welk aantal operaties kom je uit voor alle conv layers samen?
zie C2
"""

"""
## Train the model
"""

batch_size = 10 # Larger means faster training, but requires more system memory.
epochs = 250 # for now

bInitialiseWeightsFromFile = False # Set this to false if you've changed your model.

learningrate = 0.0001 if bInitialiseWeightsFromFile else 0.01

# We gebruiken alvast mean_squared_error ipv categorical_crossentropy als loss method,
# omdat straks bij opdracht D ook de afwezigheid van een cijfer een valide mogelijkheid is.
optimizer = keras.optimizers.Adam(lr=learningrate) #lr=0.01 is king
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

print("x_train.shape")
print(x_train.shape)

print("y_train.shape")
print(y_train.shape)

if (bInitialiseWeightsFromFile):
    model.load_weights("Spoiler_C_weights.h5"); # let's continue from where we ended the previous time. That should be possible if we did not change the model.
                                        # if you use the model from the spoiler, you
                                        # can avoid training-time by using "Spoiler_C_weights.h5" here.
try:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
except KeyboardInterrupt:
    print("interrupted fit by keyboard")

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", ViewTools_NN.getColoredText(255,255,0,score[1]))

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
nLastLayer = len(model.layers)-1
nLayer=nLastLayer
print("lastLayer:",nLastLayer)

baseFilenameForSave=None
x_test_flat=None
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 0, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 1, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 2, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 3, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 4, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 6, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 5, baseFilenameForSave)
ViewTools_NN.printFeatureMapsForLayer(nLayer, model, x_test_flat, x_test, 7, baseFilenameForSave)
