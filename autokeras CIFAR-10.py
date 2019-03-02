# for autokeras installation
# !pip install autokeras

from sklearn.metrics import classification_report
from keras.datasets import cifar10
import numpy as np
import autokeras as ak

TRAINING_TIMES = [
    60 * 60 * 2,  
    60 * 60 * 4,
    60 * 60 * 6,
    60 * 60 * 8]

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
 
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("Size of train dataset: ", np.shape(trainX))
print("Size of train labels: ", np.shape(trainY))
​
print("Size of test dataset: ", np.shape(testX))
print("Size of test labels: ", np.shape(testY))

for seconds in TRAINING_TIMES:
  # train our Auto-Keras model
  print("Training model for {} seconds max...".format(seconds))
  model = ak.ImageClassifier(verbose=True)
  model.fit(trainX, trainY, time_limit=seconds)
  ############################### what is the epoch?
  model.final_fit(trainX, trainY, testX, testY, retrain=True)
 
  # evaluate the Auto-Keras model
  score = model.evaluate(testX, testY)
  predictions = model.predict(testX)
  report = classification_report(testY, predictions, target_names=labelNames)
  
print(report)

model.export_autokeras_model("autokeras_exported_model")
​
