# Завдання 1:
# api: https://dummyjson.com/docs/products
# Отримати JSON-файл у Python.
# Вивести імена продуктів, які мають рейтинг 5.
# Знайти середню ціну усіх продуктів.

# Data Sources: API Files Collections
# Formats: JSON CSV XML
import requests

def task1():
    response = requests.get("https://dummyjson.com/products")

    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: {response.status_code}")
        print("Failed to retrieve data.")

    products = data['products']

    print("Продукти з рейтингом 5:")
    for product in products:
        if product['rating'] == 5:
            print("-", product['title'])

    total_price = sum(product['price'] for product in products)
    avg = total_price/len(product)

    print(f"Середня ціна усiх продуктів: {avg}")


# Завдання 2:
# https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml
# Завантажити XML.
# Вивести курс долару та фунтів.

from bs4 import BeautifulSoup

def task2():
    with open("./assets/eurofxref-daily.xml", "r") as file:
        xml_content = file.read()

    root = BeautifulSoup(xml_content, "xml")
    
    cubes = root.find_all("Cube", {"currency": True})
    
    for cube in cubes:
        currency = cube["currency"]
        rate = cube["rate"]

        if currency == "USD":
            print("Курс долара (USD):", rate)
        elif currency == "GBP":
            print("Курс фунтів (GBP):", rate)


# Завдання 3:
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
# Завантажити CSV-файл за допомогою pandas.
# Вивести імена пасажирів віком до 33 років.
# Додати новий стовпець birth_year, де визначити рік народження за віком на момент події.

import pandas as pd

def task3():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    passengers_under_33 = df[df['Age'] < 33], ['Name', 'Age']
    print("Імена пасажирів віком до 33 років:")
    print(passengers_under_33)

    df['birth_year'] = 1912 - df['Age']

    print(df[['Name', 'Age', 'birth_year']].head())



#task1()
#task2()
task3()