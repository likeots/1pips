# Руководство по проекту 1pips

## Назначение
Проект автоматизирует сравнение базового стоп-лосса и стоп-лосса, расширенного на +1 пипс, на основе тиковых данных. Он включает:

* инструменты для подготовки тиковых CSV (TrueFX и собственные выгрузки);
* симулятор `one_pip_stop_sensitivity.py`, который воспроизводит сделки SMC-адаптера и сравнивает итоговый win-rate;
* статичный дашборд в папке `site/`, который визуализирует результаты последнего запуска.

## Зависимости
* Python 3.9+
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [tqdm](https://tqdm.github.io/)
* библиотека [`smartmoneyconcepts`](https://pypi.org/project/smartmoneyconcepts/) и локальный адаптер `smc_adapter_josh.py`

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy tqdm smartmoneyconcepts
```

## Подготовка данных
### 1. Конвертация выгрузки TrueFX
TrueFX предоставляет CSV вида `EUR/USD,20250701 00:00:00.953,1.17862,1.17866`.

```bash
python convert_truefx.py EURUSD-2025-09.csv
```

Скрипт создаст файл `EURUSD-2025-09_converted.csv` c колонками `time,bid,ask` и корректной таймзоной.

### 2. Конвертация собственного CSV
Если у вас уже есть CSV `time,bid,ask`, при необходимости сформируйте минутные свечи:

```bash
python convert_ticks.py
```

Скрипт читает `ticks.csv`, агрегирует mid-цену до минутных OHLC и сохраняет результат в `candles_1m.csv`.

## Запуск симулятора
Основной сценарий — `one_pip_stop_sensitivity.py`.

```bash
python one_pip_stop_sensitivity.py \
  --csv EURUSD-2025-09_converted.csv \
  --pip-size 0.0001 \
  --smc-swing-lookback 50 \
  --retest-tolerance-pips 0.2 \
  --tp-mode structure \
  --min-rr 1.0 \
  --min-sl-pips 1.0 \
  --min-tp-pips 1.0
```

Ключевые параметры:

* `--pip-size` — размер пипса для инструмента (EURUSD = 0.0001, XAUUSD = 0.1 и т.д.).
* `--tp-mode structure|symmetric` — использовать тейк-профит адаптера или симметричный тейк с заданным RR.
* `--min-rr`, `--min-sl-pips`, `--min-tp-pips` — фильтры по R/R, риску и вознаграждению в пипсах.
* Параметры `--smc-sessions`, `--smc-fvg`, `--tf` оставлены для совместимости со старыми пайплайнами.

### Выходные файлы
После завершения симулятор сохранит:

* `one_pip_results_detailed.csv` — подробный список сделок (entry time, цены, исходы базового и +1 пипс стопов).
* `site/data/one_pip_summary.json` — агрегированные метрики (win-rate, McNemar, фильтры, разбивка по направлению).
* `smc_lib_debug.csv` — журнал причин, по которым адаптер мог отбраковать сделки.

## Обновление и просмотр дашборда
Дашборд в `site/` читает `site/data/one_pip_summary.json`. Чтобы посмотреть обновлённые результаты:

```bash
cd site
python -m http.server 8000
```

Откройте в браузере `http://localhost:8000` — страница отобразит win-rate базового стопа и стопа с добавленным пипсом, параметры фильтрации и разбивку по направлениям.

## Советы по воспроизведению
* Убедитесь, что `smc_adapter_josh.py` соответствует версии `smartmoneyconcepts`, используемой в проекте.
* Если симулятор сообщает, что все сделки отфильтрованы, попробуйте снизить `--min-rr` или переключиться на `--tp-mode symmetric`.
* Для обработки больших CSV используйте SSD и достаточно свободной памяти; `convert_truefx.py` уже записывает результат чанками, чтобы снизить нагрузку.
