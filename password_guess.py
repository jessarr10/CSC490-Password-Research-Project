import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#adding test data and spliting data
df = pd.read_csv("data.csv")

x = df.iloc[:25000, :-1].values
y = df.iloc[:25000, -1].values

encoder = OneHotEncoder(sparse_output=False)
x = encoder.fit_transform(x.reshape(-1, 1))
y = tf.keras.utils.to_categorical(y, num_classes=3)

#split model into testing and training data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=.5, random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=.4, random_state=0)

#build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#evaluate with training data
model.evaluate(x_train, y_train)

#train data - want accuracy to increase, loss to decrease
model.fit(x_train, y_train, batch_size=16, epochs=10)

#evaluate with test data
model.evaluate(x_test, y_test)
