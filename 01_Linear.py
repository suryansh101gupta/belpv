import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('BostonHousing.csv')
df.head()

df.describe()

df.info()

df.corr()

sns.heatmap(df.corr(), annot=False)

df.isnull().sum()

df['rm'].fillna(df['rm'].mean(), inplace=True)
df.isnull().sum()

x = df.drop('medv', axis=1)
y = df['medv']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))      
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))      

model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])

history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train Loss', 'Validation Loss'])

metrics = model.evaluate(x_test, y_test)
metrics

y_pred = model.predict(x_test)

y_test = np.array(y_test).reshape(-1, 1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()