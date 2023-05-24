from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil

# Dependencies:
# tensorflow        (pypi, pip install tensorflow)
# matplotlib        (pypi, pip install matplotlib)
#
# How to use:
# 1. Change the DATASET_PATH variable to your dataset path
# 2. Change the SPLITTED_PATH variable to your splitted dataset path (SPLITTED_PATH folder will be created)
# 3. Change the RATIO variable to your preferred split ratio
# 4. Change the PREDICTION_PATH variable to the predition dataset path (for testing the model)
# 5. Run the script


def splitDataset(originalPath, ratio, savedPath):
    # Check if the originalPath folder exist or not
    if not os.path.exists(originalPath):
        raise Exception("OriginalPath folder not found, please check the path")

    # Check if the savedPath folder exist or not
    if not os.path.exists(savedPath):
        os.makedirs(savedPath)
    else:
        raise Exception("SavedPath folder already exist, please choose another folder")
    
    # Create train and test folder
    trainPath = os.path.join(savedPath, "train")
    testPath = os.path.join(savedPath, "test")
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
    if not os.path.exists(testPath):
        os.makedirs(testPath)

    # Get all the class name from the originalPath folder and create the folder in train and test folder
    classes = os.listdir(originalPath)
    for className in classes:
        trainClassPath = os.path.join(trainPath, className)
        testClassPath = os.path.join(testPath, className)
        if not os.path.exists(trainClassPath):
            os.makedirs(trainClassPath)
        if not os.path.exists(testClassPath):
            os.makedirs(testClassPath)

    # Split the dataset and keep the copy in the originalPath folder
    for className in classes:
        classPath = os.path.join(originalPath, className)
        trainClassPath = os.path.join(trainPath, className)
        testClassPath = os.path.join(testPath, className)
        images = os.listdir(classPath)
        trainImages = images[:int(len(images)*ratio)]
        testImages = images[int(len(images)*ratio):]
        for image in trainImages:
            shutil.copy(os.path.join(classPath, image), os.path.join(trainClassPath, image))
        for image in testImages:
            shutil.copy(os.path.join(classPath, image), os.path.join(testClassPath, image))

    return trainPath, testPath

def trainValGen(trainPath, valPath):
    # Do data augmentation for train data and prepare validation data
    trainDatagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    valDatagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        )
    
    trainGenerator = trainDatagen.flow_from_directory(
        directory=trainPath,
        batch_size=25,
        class_mode='categorical',
        target_size=(300, 300)
        )
    
    valGenerator = valDatagen.flow_from_directory(
        directory=valPath,
        batch_size=25,
        class_mode='categorical',
        target_size=(300, 300)
        )

    return trainGenerator, valGenerator

def createModel(optimizer, loss):
    # Create the model
    model = models.Sequential([ 
        layers.Conv2D(128, (3,3), activation='relu', input_shape=(300, 300, 3)),
        layers.MaxPool2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
        ) 
    
    return model

def predGen(predPath):
    predDatagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        )
    
    predGenerator = predDatagen.flow_from_directory(
        directory=predPath,
        class_mode=None,
        target_size=(300, 300)
        )
    
    return predGenerator

def plotHistory(history):
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs=range(len(acc))

    plt.plot(epochs, acc, 'r', "Train Acc")
    plt.plot(epochs, val_acc, 'b', "Val Acc")
    plt.title('TrainVal accuracy')
    plt.show()

    plt.plot(epochs, loss, 'r', "Train Loss")
    plt.plot(epochs, val_loss, 'b', "Val Loss")
    plt.show()

# Variable for dataset, splitted = dataset with train and test folder
DATASET_PATH = "C:/Users/Administrator/Desktop/Capstone/Data/AttractionDataset/"
SPLITTED_PATH = "C:/Users/Administrator/Desktop/Capstone/Data/AttractionDataset-Splitted/"
PREDICTION_PATH = ""
RATIO = 0.85

# Prepare the dataset
trainPath, valPath = splitDataset(DATASET_PATH, RATIO, SPLITTED_PATH)
trainGen, valGen = trainValGen(trainPath, valPath)

# Variable for the model
LEARNING_RATE = 0.001
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS = losses.CategoricalCrossentropy()
SAVED_MODEL_PATH = "C:/Users/Administrator/Desktop/Capstone/Model/Test.h5"

# Create the model
model = createModel(OPTIMIZER, LOSS)

# Train the model
#history = model.fit(
    # trainGen, 
    # epochs=10, 
    # validation_data=valGen, 
    # verbose=1
    # )

# Save and plot the model
#model.save(SAVED_MODEL_PATH)
#plotHistory(history)

# This code is for testing the prediction using saved model

# model = tf.keras.saving.load_model("C:/Users/Administrator/Desktop/Capstone/Model/Test.h5")
# res = model.predict(predGen)
# lab = list(trainGen.class_indices.keys())
# sel = res.argmax(axis=1)
# print(res)
# print(predGen.filenames)
# print(f"Predicted class: {lab[sel[0]]}, with probability {res[0][sel[0]]}")
