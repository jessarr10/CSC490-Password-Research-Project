import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf

df = pd.read_csv("practice_diabetes.csv")

#getting data
x = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

#scale data
scaler = StandardScaler()
x = scaler.fit_transform(x)

#random sample
over = RandomOverSampler()
x, y = over.fit_resample(x, y)
data = np.hstack((x, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

#splitting data into training data and test data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=.4, random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=.5, random_state=0)

#build model
#layer of 16 neurons, activation function - relu
model = tf.keras.Sequential([
                                tf.keras.layers.Dense(16, activation='relu'),
                                tf.keras.layers.Dense(16, activation='relu'),
                                tf.keras.layers.Dense(1, activation='sigmoid')  #binary activation
])

#compile model (choose optimizer algorithm)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

#evaluate with training data
model.evaluate(x_train, y_train)

#train data
#want accuracy to increase, loss to decrease
model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_valid, y_valid))

#evaluate with test data
model.evaluate(x_test, y_test)
