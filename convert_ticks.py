import pandas as pd

# Загружаем CSV с заголовками
df = pd.read_csv("ticks.csv")

# Если у тебя в CSV нет заголовков, то замени строчку выше на:
# df = pd.read_csv("ticks.csv", header=None, names=["time", "bid", "ask"])

# Преобразуем в datetime (с учётом формата ISO 8601 и таймзоны)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

# Убираем строки, которые не смогли конвертироваться
df = df.dropna(subset=["time"])

# Ставим индекс времени
df.set_index("time", inplace=True)

# Создаём колонку "mid" (средняя цена между bid и ask)
df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2

# Агрегация по минутам (Open, High, Low, Close)
candles = df["mid"].resample("1T").ohlc()

# Сохраняем
candles.to_csv("candles_1m.csv", float_format="%.5f")

print("Готово! Результат сохранён в candles_1m.csv")
