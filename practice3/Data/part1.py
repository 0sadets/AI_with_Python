import pandas as pd

# 1. Створити DataFrame з цих даних та конвертувати колонку OrderDate у тип datetime. 
df = pd.read_csv('orders.csv')

def DataFrameCreate():
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])

    print(df)
    print(df.dtypes) # output: OrderDate    datetime64[ns]
#DataFrameCreate()
#------------------------------------------------------------------------------------------------------------------
# 2. Додати новий стовпець ТotalAmount = Quantity * Price. 
def addТotalAmount():
    df['TotalAmount'] = df['Quantity'] * df['Price']

    print(df) # output:  0           1     Olivia Davis    Basketball       Sports         1  6026.24 2024-12-29      6026.24

#addТotalAmount()
#------------------------------------------------------------------------------------------------------------------
# 3. Вивести: 
# а. Сумарний дохід магазину. 
# b. Середнє значення TotalAmount. 
# с. Кількість замовлень по кожному клієнту. 

def PrintData():
    total_income = df['TotalAmount'].sum()
    print(f"Сумарний дохід магазину: {total_income}") #output: Сумарний дохід магазину: 138416560.29000002

    average_total = df['TotalAmount'].mean()
    print(f"Середнє значення TotalAmount: {average_total}") #output: Середнє значення TotalAmount: 27683.312058000003

    orders_per_customer = df.groupby('Customer')['OrderID'].count() 
    print("Кількість замовлень по кожному клієнту:")
    print(orders_per_customer) #output:Alice Johnson      501 Chris Lee          471 Emma Wilson        511

#PrintData()

#------------------------------------------------------------------------------------------------------------------
# 4. Вивести замовлення, в яких сума покупки перевищує 500. 
def task4():
    high_value_orders = df[df['TotalAmount'] > 500]

    print("Замовлення з сумою покупки більше 500:")
    print(high_value_orders)

#addТotalAmount()
#task4()

#------------------------------------------------------------------------------------------------------------------
#5. Відсортувати таблицю за OrderDate у зворотному порядку. 

def SortOrderDate():
    df_sorted = df.sort_values(by='OrderDate', ascending=False)
    print("Таблиця після сортування:")
    print(df_sorted)

#SortOrderDate()

#------------------------------------------------------------------------------------------------------------------
#6. Вивести всі замовлення, зроблені у період з 5 по 10 червня включно. 

def printBuData():
    start_date = '2024-06-05'
    end_date = '2024-06-10'

    orders_in_june = df[(df['OrderDate'] >= start_date) & (df['OrderDate'] <= end_date)]

    print("Замовлення з 5 по 10 червня:")
    print(orders_in_june)

#printBuData()

#------------------------------------------------------------------------------------------------------------------
#8. Вивести ТОП-3 клієнтів за загальною сумою покупок (TotalAmount).

def top3():
    customer_totals = df.groupby('Customer')['TotalAmount'].sum()

    top_3_customers = customer_totals.sort_values(ascending=False).head(3)

    print("Топ-3 клієнтів за загальною сумою покупок:")
    print(top_3_customers)

addТotalAmount()
top3()
#output:
# Топ-3 клієнтів за загальною сумою покупок:
# Customer
# Jane Smith     15163327.55
# Noah Garcia    14508423.13
# John Doe       14381635.98

#------------------------------------------------------------------------------------------------------------------
