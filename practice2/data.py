# Завдання 1:
# Побудувати модель машинного навчання використовуючи лінійну регресію, яка
# передбачає обсяг споживаної електроенергії в районі міста на основі таких даних:
# • temperature - температура повітря (°C)
# • humidity - вологість (%)
# • hour - година доби (0–23)
# • is_weekend: чи поточний день є вихідним (0, 1)
# • consumption - електроспоживання (кВт·год)

# Файл .csv з набором даних надається або можна згенерувати з ШІ.
# Побудувати графік «Справжня vs Прогнозована ціна» та визначити % помилки.

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def task1():
    df = pd.read_csv("energy_usage.csv")
    print(df.head())

    # temperature,humidity,hour,is_weekend,consumption
    # Вибір ознак та цільової змінної
    X = df[['temperature', 'humidity', 'hour', 'is_weekend']] # features
    y = df['consumption']                                     # target

    # Розділення на train/test
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # Створення моделi лінійної регресії і її навчання  на тренувальних даних
    model = LinearRegression()
    model.fit(X_training, y_training)

    # Прогноз та оцінка
    your_electricity = pd.DataFrame([{
        'temperature':15,  # температура
        'humidity':65,     # вологість
        'hour':16,         # година доби
        'is_weekend':0     # чи вихідні
    }])

    predicted_el = model.predict(your_electricity)
    print(f"Прогнозований обсяг споживання електроенергії: {predicted_el[0]:,.2f} кВт·год")

    y_pred = model.predict(X_test)

    # Оцінка точності моделі
    # MAPE показує середню відносну помилку в %
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"MAPE: {mape:.2f}%")

    # Візуалізація: Справжній vs Прогнозований обсяг
    plt.scatter(y_test, y_pred)
    plt.xlabel("Справжній обсяг")
    plt.ylabel("Прогнозований обсяг")
    plt.title("Справжній vs Прогнозований обсяг")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
    plt.show()
# -----------------------------------------------------------------------------------------------------
# *Завдання 2:
# Додати до району такі властивості:
# • season - пора року (winter, summer)
# • district_type - тип району (industrial, residential)
# Використати кодування категоріальних ознак (текстових значень) одним із
# доступних методів (OneHotEncoder).

def task2():
    df = pd.read_csv("energy_usage_plus.csv")  
    print(df.head())

    # temperature,humidity,season,hour,district_type,is_weekend,consumption
    X = df[[
        'temperature', 
        'humidity', 
        'season', 
        'hour',
        'district_type', 
        'is_weekend'
        ]]
    
    # categorical_features = ['season', 'district_type']
    # numeric_features = ['temperature', 'humidity', 'hour', 'is_weekend']

    y = df['consumption']

    # кодування категоріальних змінних season, district_type
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat = encoder.fit_transform(X[['season', 'district_type']])
    # oбєднання числових і закодованих ознак
    X_num = X[['temperature', 'humidity', 'hour', 'is_weekend']].values
    X_encoded = np.hstack((X_num, X_cat))

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=12)


    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"MAPE: {mape:.2f}%")

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Справжній обсяг")
    plt.ylabel("Прогнозований обсяг")
    plt.title("Справжній vs Прогнозований обсяг")
    plt.grid(True)
    plt.show()

#task1()
task2()