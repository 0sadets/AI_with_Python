import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

# Завдання 1:
# Зобразити графік функції:
# f(x) = x^2*sin(x) x E [-10,10 ] 
def task1():
    x = np.linspace(-10, 10, 500)
    y = pow(x,2)*np.sin(x)

    plt.plot(x, y)
    plt.title("Графік функції x^2*sin(x)")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid(True)
    plt.show()
#task1()
# --------------------------------------------------------------------------------------------
# Завдання 2:
# Згенерувати масив 1000 випадкових чисел з нормальним розподілом (середнє = 5,
# σ = 2) і побудувати для них гістограму.
def task2():
    data = np.random.normal(loc=5, scale=2, size=1000)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title('Гістограма нормального розподілу')
    plt.xlabel('Значення')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()
#task2()
# --------------------------------------------------------------------------------------------
# Завдання 3:
# Створити свою кругову діаграму — обери 4-5 улюблених хобі і задай їх «частки» у
# своєму житті :)
def task3():
    labels = ['reading', 'cooking', 'friends', 'teeth_curing']
    sizes = [40, 15, 25, 20]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Hobbies")
    plt.axis('equal') 
    plt.show()

#task3()
# --------------------------------------------------------------------------------------------
# Завдання 4:
# Створити «box-plot» для 4 типів фруктів, де дані — це маса 100 фруктів кожного
# виду (рандомно з нормального розподілу).
def task4():
    fruit_names = ['Apple', 'Banana', 'Orange', 'Pear']
    np.random.seed(0)
    fruit_weights = [
    np.random.normal(loc=150, scale=15, size=100),  
    np.random.normal(loc=120, scale=10, size=100), 
    np.random.normal(loc=160, scale=20, size=100), 
    np.random.normal(loc=140, scale=12, size=100)   
    ]
    plt.figure(figsize=(10, 6))
    plt.boxplot(fruit_weights, labels=fruit_names)
    plt.title('Box-plot маси фруктів')
    plt.ylabel('Маса (г)')
    plt.grid(True)
    plt.show()
#task4()
# --------------------------------------------------------------------------------------------
# *Завдання 5:
# Згенерувати 100 точок для x та y з рівномірного розподілу на [0, 1].
# Побудувати точкову діаграму та задати:
# • зелений колір;
# • прозорість (alpha=0.6);
# • назви осей.
def task5():
    np.random.seed(0) 
    x = np.random.uniform(0, 1, 100)
    y = np.random.uniform(0, 1, 100)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='green', alpha=0.6)
    plt.title('Точкова діаграма випадкових точок')
    plt.xlabel('Значення X')
    plt.ylabel('Значення Y')
    plt.grid(True)
    plt.show()
#task5()
# --------------------------------------------------------------------------------------------
#*Завдання 6:
# Побудувати графіки трьох функцій на одному графіку:
# f(x) = sin(x)
# g(x) = cos(x)
# h(x) = sin(x) + cos(x)
# Зробити:
# • різні кольори;
# • легенду;
# • сітку;
# • підписи осей та заголовок.
def task6():
    x = np.linspace(0, 2 * np.pi, 500)
    f = np.sin(x)
    g = np.cos(x)
    h = f + g

    plt.figure(figsize=(10, 6))
    plt.plot(x, f, label='sin(x)', color='blue')
    plt.plot(x, g, label='cos(x)', color='red')
    plt.plot(x, h, label='sin(x) + cos(x)', color='green')
    plt.title('Графіки тригонометричних функцій')
    plt.xlabel('x')
    plt.ylabel('Значення функції')
    plt.legend()
    plt.grid(True)
    plt.show()
task6()