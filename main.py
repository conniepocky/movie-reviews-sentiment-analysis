import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2" 
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential([
    hub_layer, #first layer, converts text to vector
    tf.keras.layers.Dense(16, activation='relu'),#16 hidden neurons
    tf.keras.layers.Dense(1) #one output neuron
])

model.summary()

model.compile(optimizer="adam",
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #as this is a binary classification problem, binary crossentropy is used
                metrics=["accuracy"])

history = model.fit(train_data.shuffle(10000).batch(512), #shuffle the data and train in batches of 512
                    epochs=10,
                    validation_data=validation_data.batch(512), #validation data is used to check if the model is overfitting
                    verbose=1) 

results = model.evaluate(test_data.batch(512), verbose=2) #evaluate the model on the test data

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value)) #lower loss the better, accuracy is the percentage of correct predictions

#test predictions
  
test_simple_reviews = [
  "The movie is the best I've ever seen!",
  "The movie was amazing!",
  "The movie was okay.",
  "I hated the movie it was awful.",
  "The movie was terrible... total waste of money..."
]

simple_review_predictions = model.predict(test_simple_reviews)

for review, prediction in zip(test_simple_reviews, simple_review_predictions):
    print("Review: ", review, "Prediction: ", "Positive" if prediction > 0 else  "Negative")

test_predictions = model.predict(test_data.batch(512))

y = np.array(test_predictions).flatten()

pos_neg = [0, 0]
labels = ["Positive", "Negative"]
colours = ["darkseagreen", "crimson"]

for prediction in y:
    if prediction > 0:
        pos_neg[0] += prediction
    else:
        pos_neg[1] -= prediction

plt.pie(pos_neg, labels=labels, colors=colours, autopct='%1.1f%%', startangle=90)

plt.show()