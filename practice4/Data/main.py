# Створити ML модель для передбачення тривалості поїздки, включаючи нічний часу. 
# Використати поліноміальну регресію.
 
# Візуалізувати графік: фактичні точки + крива регресії.
 
# Оцінити якість моделі визначивши коефіцієнти помилки (MAE, MSE).
# Визначити тривалість поїздки в: 10:30
# 00:00, 02:40
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# читажмо дані
df = pd.read_csv("practice4/rides.csv")
# print(df.head())

# визначаємо змінні
X = df[['Hour']]
Y = df[['Duration']]

# 3. Створюємо модель: Поліноміальна регресія ступеня 2
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 4. Навчаємо модель на всіх даних
model.fit(X, Y)

# 5. Генеруємо тестові дані в межах [0, 24]
X_test = np.linspace(0, 24, 10000).reshape(-1, 1)
Y_test = model.predict(X_test)


# 7. Побудова графіку
plt.figure(figsize=(10, 6))
plt.plot(X, Y, label='Real function (з шумом)', color='blue')
plt.plot(X_test, Y_test, label='Predicted function (Poly degree 2)', color='red', linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Real vs Predicted Function (0 to 24)')
plt.legend()
plt.grid(True)
plt.show()

# 8. Передбачення для конкретної точки
predict_hours = np.array([[10.5], [0.0], [2.67]])  
predicted_values = model.predict(predict_hours)

for h, val in zip(['10:30', '00:00', '02:40'], predicted_values):
    print(f"Прогноз тривалості поїздки о {h}: {val[0]:.2f} хв")

# оцінка якості моделі
y_train_pred = model.predict(X)
mae = mean_absolute_error(Y, y_train_pred)
mse = mean_squared_error(Y, y_train_pred)
print(f"MAE (Середня абсолютна похибка): {mae:.4f}")
print(f"MSE (Середньоквадратична похибка): {mse:.4f}")
