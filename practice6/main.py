# Завдання 1:
# Автомобіль на певній ділянці дороги був протестований при різних швидкостях.
# Для кожної швидкості було виміряно витрати пального (літрів на 100 км).
# Завдання — побудувати модель, яка наближає цю залежність та дозволяє робити
# прогнози.
# Використати поліноміальну регресію, визначити оптимальний ступінь.
# Порівняти точність моделей за метрикою MSE, MAE.
# Обрати найкращу модель і передбачити витрати пального на швидкості 35, 95, 140
# км/год.
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("practice6/fuel_consumption_vs_speed.csv")
X = df[['speed_kmh']]
Y = df['fuel_consumption_l_per_100km']

degree = 3
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X, Y)

X_test = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
Y_test = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Фактичні дані')
plt.plot(X_test, Y_test, color='red', linestyle='--', label=f'Поліном (ступінь {degree})')
plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрати пального (л/100 км)')
plt.title('Витрати пального від швидкості')
plt.legend()
plt.grid(True)
plt.show()

predict_speeds = np.array([[35], [95], [140]])
predicted_consumption = model.predict(predict_speeds)

for speed, consumption in zip([35, 95, 140], predicted_consumption):
    print(f"Прогноз витрат пального на {speed} км/год: {consumption:.2f} л/100 км")


y_train_pred = model.predict(X)
mae = mean_absolute_error(Y, y_train_pred)
mse = mean_squared_error(Y, y_train_pred)
print(f"MAE (Середня абсолютна похибка): {mae:.4f}")
print(f"MSE (Середньоквадратична похибка): {mse:.4f}")