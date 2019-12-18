import tensorflow as tf
import numpy as np
import shutil

shutil.rmtree("outdir", ignore_errors=True)

def train_input_fn():
  features = {"sq_footage":[ 1000,    2000,    3000,    1000,  2000,  3000],
              "type":      ["house", "house", "house", "apt", "apt", "apt"]}
  label =                  [ 500,     1000,    1500,    700,   1300,  1900]
  return features, label

featcols = [
            tf.feature_column.numeric_column("sq_footage"),
            tf.feature_column.categorical_column_with_vocabulary_list("type", ["house", "apt"])
]

model = tf.estimator.LinearRegressor(featcols, "outdir")
model.train(train_input_fn, steps=2000)



def predict_input_fn():
  features = {"sq_footage":[ 1500,   1500],
              "type":      ["house", "apt"]}
  return features

predictions = model.predict(predict_input_fn)
print(next(predictions))
print(next(predictions))
