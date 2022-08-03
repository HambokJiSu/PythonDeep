import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("gpascore.csv")
#   데이터 전처리
data = data.dropna()   #   빈 칸 데이터 삭제
#data = data.fillna()   #   빈 칸에 원하는 데이터 삽입

arrX = []
for i, row in data.iterrows():
    arrX.append([row["gre"], row["gpa"], row["rank"]])

arrY = data["admit"].values

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"), #   sigmoid는 0~1 사이의 값을 뱉음
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(np.array(arrX), np.array(arrY), epochs=1000) #   fit(numpy array X데이터, numpy array y데이터, epochs=학습횟수)

epValue = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(epValue)