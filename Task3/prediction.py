import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

# -------------START_TEST----------------#
# Перевірка наявності файлу
if not os.path.exists("data.csv"):
    raise FileNotFoundError("Can`t found .csv file!")

# Завантаження даних
df = pd.read_csv("data.csv", names=['Datetime', 'Temp'], skiprows=10)

# Перевірка, чи є дані в датафреймі
if df.empty:
    raise ValueError("Can`t find data in .csv file!")
# -------------END_TEST----------------#

# Витяг дати та часу
df[['Date', 'Time']] = df['Datetime'].str.extract(r'(\d{8})T(\d{4})')

# Перетворення 'Date' у формат datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# -------------START_TEST----------------#
# Перевірка наявності стовпця 'Datetime'
if 'Datetime' not in df.columns:
    raise KeyError("There is no \'Datatime\' column")

# Перевірка коректності перетворення дати
try:
    df['Date'] = pd.to_datetime(df['Datetime'].str.extract(
        r'(\d{8})T(\d{4})')[0], format='%Y%m%d')
except Exception as e:
    raise ValueError(f"Error while reformate the date: {e}")
# -------------END_TEST----------------#

# Фільтрація даних між 2022-04-21 та 2025-04-21
train_start_date = pd.to_datetime('2022-04-21')
train_end_date = pd.to_datetime('2025-04-21')

# Обчислення середньої температури по кожному дню
daily_avg = df.groupby('Date')['Temp'].mean().reset_index()

# Перейменування колонок
daily_avg.columns = ['Date', 'AvgTemp']

# Фільтруємо дані для навчання (2022-04-21 по 2025-04-21)
train_data = daily_avg[(daily_avg['Date'] >= train_start_date) & (
    daily_avg['Date'] <= train_end_date)]

# Створюємо індекс дня (кількість днів від початку)
train_data['DayIndex'] = (
    train_data['Date'] - train_data['Date'].min()).dt.days

# Додаткові ознаки
train_data['DayOfYear'] = train_data['Date'].dt.dayofyear  # день року
train_data['Month'] = train_data['Date'].dt.month         # місяць
train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek  # день тижня

# Вхід (X) і ціль (y)
X_train = train_data[['DayOfYear', 'Month', 'DayOfWeek']]  # додаткові ознаки
# середня температура
y_train = train_data['AvgTemp']

# Створення моделі Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозування температури для наступного року
# Початок прогнозу (21 квітня 2024)
next_year_start_date = pd.to_datetime('2025-01-01')
future_X = pd.DataFrame({
    'DayOfYear': np.arange(1, 366),  # Прогноз на цілий рік
    'Month': [i % 12 + 1 for i in range(1, 366)],  # Місяці для кожного дня
    'DayOfWeek': np.random.randint(0, 7, 365)     # День тижня для кожного дня
})

# -------------START_TEST----------------#
# Перевірка, чи модель була навченена
if model is None:
    raise ValueError("Model has not been created or trained!")

# Перевірка прогнозування на основі навчальних даних
try:
    y_pred = model.predict(X_train)
except Exception as e:
    raise ValueError(f"Error in forecasting: {e}")

# Перевірка, чи має прогноз ненульову довжину
if len(y_pred) == 0:
    raise ValueError("Forecasting doesn`t get any results!")
# -------------END_TEST----------------#

# Прогноз температури для наступного року
future_y = model.predict(future_X)

# Обчислення стандартної помилки для прогнозів
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
std_error = np.sqrt(mse)

# Розрахунок довірчого інтервалу
confidence_interval = 1.96 * std_error  # для 95% довірчого інтервалу

# Створюємо верхню та нижню межу інтервалу
future_X['PredictedTemp'] = future_y
future_X['LowerBound'] = future_y - confidence_interval
future_X['UpperBound'] = future_y + confidence_interval

# Додаємо дату для зручності
future_X['Date'] = next_year_start_date + \
    pd.to_timedelta(future_X['DayOfYear'] - 1, unit='D')

# Обмежуємо кількість знаків після коми до однієї
future_X['PredictedTemp'] = future_X['PredictedTemp'].round(1)
future_X['LowerBound'] = future_X['LowerBound'].round(1)
future_X['UpperBound'] = future_X['UpperBound'].round(1)

# Виведення всіх рядків
pd.set_option('display.max_rows', None)  # Вивести всі рядки
# Вивести числа з 1 знаком після коми
pd.set_option('display.float_format', '{:.1f}'.format)

# Створення таблиці
result_df = future_X[['Date', 'PredictedTemp', 'LowerBound', 'UpperBound']]

# Виведення таблиці
print(result_df)

# Графік
plt.figure(figsize=(10, 6))

# Графік прогнозованих температур
plt.plot(future_X['Date'], future_X['PredictedTemp'],
         label='Прогнозована температура', color='blue')

# Графік довірчого інтервалу
plt.fill_between(future_X['Date'], future_X['LowerBound'], future_X['UpperBound'],
                 color='lightblue', alpha=0.5, label='95% довірчий інтервал')

# Заголовок та підписи
plt.title('Прогноз температури на наступний рік з 01.01.2025')
plt.xlabel('Дата')
plt.ylabel('Температура (°C)')
plt.xticks(rotation=45)
plt.legend()

# Показати графік
plt.tight_layout()
plt.show()

# Якщо хочеш зберегти таблицю у файл (наприклад, CSV):
# result_df.to_csv('temperature_forecast.csv', index=False)
