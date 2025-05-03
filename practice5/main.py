# Завдання 1:
# Написати модель машинного навчання для передбачення, чи буде студент 
# прийнятий на стажування в SoftServe, використовуючи наступні критерії:
# • Досвід (у роках) — Experience
# • Середній бал в Mystat (за 12-бальною шкалою) — Grade
# • Рівень англійської мови (1, 2, 3…) — EnglishLevel
# • Вік — Age
# • Бали за вступний тест (максимум 1000) — EntryTestScore
# • Чи був прийнятий (1 — так, 0 — ні) — Accepted
# Для вирішення використати логістичну регресію.
# Побудувати графік ймовірності прийняття від рівня англійської та вступного тесту.
# ! Завантажити посилання на репозиторій.
# *Завдання 2:
# Зберігати рівень англійської як значення: Elementary, Pre-Intermediate, Intermediate, 
# Upper-Intermediate, Advanced і т.д
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

def numVersion():
    df = pd.read_csv('practice5/internship_candidates_final_numeric.csv')
    task1(df)

def textVersion():
    df = pd.read_csv('practice5/internship_candidates_cefr_final.csv')
    encoder = OrdinalEncoder(categories=[['Elementary', 'Pre-Intermediate', 'Intermediate', 'Upper-Intermediate', 'Advanced']])
    df['EnglishLevel'] = encoder.fit_transform(df[['EnglishLevel']])
    task1(df)

def task1(df):
    # 1. Завантаження даних
    # df = pd.read_csv('practice5/internship_candidates_final_numeric.csv')

    # 2. Вхідні та цільові змінні
    X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
    y = df['Accepted']

    # 3. Розділення
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Побудова моделі
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Передбачення
    y_pred = model.predict(X_test)

    # 6. Візуалізація
    plt.scatter(X_test['EnglishLevel'], X_test['EntryTestScore'], c=y_pred, cmap='coolwarm', s=100, edgecolor='k')
    plt.title("Ймовірність прийняття")
    plt.xlabel("English Level")
    plt.ylabel("Entry Test Score")
    plt.colorbar(label="Прийнятий = 1 / Нi = 0")
    # plt.grid(True)
    plt.show()

    # 6. Оцінка
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



numVersion()
textVersion()