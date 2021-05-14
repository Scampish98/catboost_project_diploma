# Морфологический разбор текстов на дореволюционном русском языке методом машинного обучения с помощью Catboost #

## Модели ##

**Параметры моделей**:
  * само слово
  * несколько слов перед ним в предложении (количество указывается в конфигурационном файле)
  * несколько слов после него в предложении (количество указывается в конфигурационном файле)
  * начальная форма слова
  * уже вычисленные атрибуты (опционально, указывается в конфигурационном файле) 


Для каждого атрибута будет обучена своя Catboost модель. 

### Версия 0 ###
**Файл**: `src/lib/models/sequential_learning_model.py`

**Класс**: `SequentialLearningModel`

Обучение Catboost моделей производится по возрастанию номеров атрибутов (идентификаторы в БД).

### Версия 1 ###
**Файл**: `src/lib/models/sequential_learning_based_on_attribute_tree_model.py`

**Класс**: `SequentialLearningBasedOnAttributeTreeModel`

Эта модель использует дерево атрибутов. 
Сначала обучаем для первого атрибута (часть речи).
Затем дробим данные на группы по разным значениям 
рассмотренного атрибута.
Для каждой группы обучаем модель для нового атрибута 
(и так далее в рекурсии).
Т.е. обучение Catboost моделей атрибутов происходит в порядке
обхода в глубину дерева атрибутов.

### Версия 2 ###

**Файл**: `src/lib/models/sequential_learning_model.py`

**Класс**: `ReversedSequentialLearningModel`


То же, что и модель версии 0, только атрибуты рассматриваем в порядке убывания номеров.


## Конфигурационный файл ##

```yaml
catboost_params:                 # Параметры для catboost (описаны тут https://catboost.ai/docs)
  iterations: 100                # Количество итераций для обучения модели
  learning_rate: 0.1             # Скорость обучения модели
  loss_function: MultiClass      # Функция потерь
  task_type: CPU                 # На чем запускать (CPU/GPU). На GPU точность моделей сильно проседает.
model_type: classifier           # Тип модели (была попытка применить CatBoostRegressor, но там все стало очень плохо с точностью)
model_version: 0                 # Версия модели (0, 1, 2). Описание выше.
number_words_after: 5            # Сколько слов в предложении перед рассматриваемым использовать в качестве параметров (0-5)
number_words_before: 5           # Сколько слов в предложении после рассматриваемого использовать в качестве параметров (0-5)
refit_model: false               # Переобучить модель (по-умолчанию продолжает с момента, на котором обучение было остановленно)
logging_level: stats             # Уровень логирования (silent: ничего, debug: python, catboost - debug, info: python, catboost - info, stats: python - info, catboost - verbose, error: python - error, catboost - silent)
smart_split: true                # Разбиение данных на обучающую и тестовую выборки (для каждого атрибута независимо): для каждого возможного значения атрибута должны быть примеры и в обучающей (примерно 1/3) и в тестовой (примерно 2/3) выборках.
use_calculated_parameters: true  # Все предыдущие рассмотренные атрибуты участвуют в дальнейшем обучении как параметры.
use_initial_form: true           # Начальная форма является параметром для обучения.
lemmer_type: smalt_stemmer       # Способ получения начальной формы слова: empty - выдавать пустую строку в качестве начальной формы, smalt_stemmer - стеммер, перенесенный из `Stemmer_RU.php`
language_filter: true            # Иностранные слова определяются не обученной моделью, а отдельными алгоритмами (пока есть плавающая проблема в виде бана со стороны гугловых сервисов, ведется доработка)
excluded_attributes:             # Часть атрибутов можно исключить из выборки (104-109 из старого дерева атрибутов)
  - 105
  - 106
  - 107
  - 108
  - 109
  - 110
excluded_attribute_values:       # Слова, у которых среди значений атрибутов есть значения из списка, исключаются из обучения (369-400 из старого дерева атрибутов).
  - 370
  - 371
  - 372
  - 373
  - 374
  - 375
  - 376
  - 377
  - 378
  - 379
  - 380
  - 381
  - 382
  - 383
  - 384
  - 385
  - 386
  - 387
  - 388
  - 389
  - 390
  - 391
  - 392
  - 393
  - 394
  - 395
  - 396
  - 397
  - 398
  - 399
  - 400
  - 401
```

## Запуск ##

Все файлы для запуска лежат в корне `src/`.

### Обучение модели ###
**Файл**: `src/train.py`

**Параметры командной строки**:
  * `-c` (`--config`): путь до конфигурационного файла
  * `-m` (`--meta`): путь до файла meta.json (файл с данными о моделях)
  * `-t` (`--train-data`): путь до файла с данными для обучения
  * `-v` (`--validation-data`): путь до файла с валидационными данными 
    (опционально, используется только с опцией smart_split: false в конфигурационном файле)

**Пример запуска**:
```shell
$ python3 train.py -c ../config/config.yaml -m ../data/meta.json -t ../data/catboost_data.tsv
```
### Предсказание для текста ###
**Файл**: `src/process.py`

**Параметры командной строки**:
  * `-c` (`--config`): путь до конфигурационного файла
  * `-m` (`--meta`): путь до файла meta.json (файл с данными о моделях)
  * `--input`: путь до файла с текстом для разбора
  * `--output`: путь до файла с результатом разбора (результат будет в формате json)

**Пример запуска**
```shell
$ python3 process.py -c ../config/config.yaml -m ../data/meta.json --input ../text.txt --output ../result.json
```

**Пример результата разбора**:
```json
[
  {
    "word": "цѣлей",
    "word_id": 41,
    "sentence_id": 1,
    "paragraph_id": 7,
    "initial_form": "цѣлей",
    "result": {
      "Часть речи": "Cуществительное",
      "Аттрибуты": {
        "Разряд по значению(А)": "Неодушевленное",
        "Разряд по значению(Б)": "Нарицательное",
        "Разряд по значению(В)": "Конкретное",
        "Категория рода": "Женский",
        "Категория числа": "Множественное",
        "Категория падежа": "Родительный",
        "Типы склонения": "III склонение"
      }
    }
  },  
  {
    "word": "проповѣди",
    "word_id": 42,
    "sentence_id": 1,
    "paragraph_id": 7,
    "initial_form": "проповѣд",
    "result": {
      "Часть речи": "Cуществительное",
      "Аттрибуты": {
        "Разряд по значению(А)": "Неодушевленное",
        "Разряд по значению(Б)": "Нарицательное",
        "Разряд по значению(В)": "Конкретное",
        "Категория рода": "Женский",
        "Категория числа": "Единственное",
        "Категория падежа": "Родительный",
        "Типы склонения": "III склонение"
      }
    }
  },
  {
    "word": "и",
    "word_id": 53,
    "sentence_id": 1,
    "paragraph_id": 7,
    "initial_form": "и",
    "result": {
      "Часть речи": "Союз",
      "Аттрибуты": {
        "По составу": "Простой",
        "По употреблению": "Одиночный",
        "По синтаксической функции": "Сочинительный",
        "Аттрибуты": {
          "Разряд": "Соединительный"
        }
      }
    }
  }
]
```

### Предсказание на подготовленных данных ###
**Файл**: `src/predict.py`

**Параметры командной строки**:
  * `-c` (`--config`): путь до конфигурационного файла
  * `-m` (`--meta`): путь до файла meta.json (файл с данными о моделях)
  * `--input`: путь до файла с данными для разбора (должны быть обработаны и храниться в формате tsv)
  * `--output`: путь до файла с результатом разбора (в формате tsv)

Если в тексте (во входных данных) некоторых слов для параметров нет (например, перед этим словом в предложении меньше пяти слов),
эти слова заменяются 0.

**Пример запуска**
```shell
$ python3 predict.py -c ../config/config.yaml -m ../data/meta.json --input ../data/catboost_data.tsv --output ../result.tsv
```

**Пример входных данных**
```text
WORD	word_-5	word_-4	word_-3	word_-2	word_-1	word_1	word_2	word_3	word_4	word_5	initial_form	word_id	sentence_id	paragraph_id
общимъ	0   Мы	начали	нашу	статью	сочувствіемъ	къ	Н	И	Пирогову	общій	5	1	2
сочувствіемъ	Мы	начали	нашу	статью	общимъ	къ	Н	И	Пирогову	возбужденнымъ	сочувствіе	6	1	2
```

**Пример выходных данных**
```t
WORD	initial_form	word_-1	word_-2	word_-3	word_-4	word_-5	word_1	word_2	word_3	word_4	word_5	1	2	3	4	5	6	7	8	9	26	27	28	29	30	31	32	33	34	35	36	37	38	40	46	47	48	41	42	43	45	49	53	50	51	52	54	55	59	60	61	39	62	63	64	65	10	11	12	13	14	15	16	17	18	19	97	98	99	66	67	68	69	70	71	72	73	74	75	81	82	83	84	85	86	87	92	94	93	95	96	100	101	102	104	103	88	89	90	20	21	22	23	24	25	76	77	78	79	80	word_id	sentence_id	paragraph_id	44	56	57	58	91	105	106	107	108	109	110
общимъ	общій	статью	нашу	начали	Мы	0	сочувствіемъ	къ	Н	И	Пирогову	39	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	42	47	52	55	66	0	0	67	72	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	5	1	2	0	0	0	0	0	0	0	0	0	0	0
сочувствіемъ	сочувствіе	общимъ	статью	нашу	начали	Мы	къ	Н	И	Пирогову	возбужденнымъ	1	0	4	8	12	19	24	30	33	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	6	1	2	0	0	0	0	0	0	0	0	0	0	0
```

### Оценка точности на эталонных данных ###
**Файл**: `src/compare.py`

**Параметры командной строки**:
  * `--result-data`: путь до файла с данными для сравнения (в формате tsv)
  * `--validate-data`: путь до файла с эталонными данными (в формате tsv)

**Пример запуска**
```shell
$ python3 compare.py --result-data ../result.tsv --validate-data ../data/catboost_data.tsv
```
**Пример вывода**
```text
Name = 1, bad = 104754, good = 715145, cnt = 819899, bad percent = 12.77645173368915 good percent = 87.22354826631086
...
Name = 110, bad = 58582, good = 936, cnt = 59518, bad percent = 98.42736651097147 good percent = 1.5726334890285292
Total bad = 463554, total good = 4566268, total cnt = 5029822, total bad percent = 9.216111425016631 total good percent = 90.78388857498337
```
## Дополнительная информация ##

`Dataframe` -  самописная альтернатива `pandas.dataframe` и `numpy.array`.
В `pandas.dataframe` возникала проблема смены типа при создании из списка списков, что сказывалось на точности обучения моделей.
В `numpy.array` на больших данных возникала проблема нехватки оперативной памяти.
Дополнительно в `Dataframe` реализованы методы для более удобной работы с данными.

Данные для обучения лежат в `data/catboost_data.tsv`.
Конфигурационный файл модели лежит в `config/config.yaml`.
Данные об имеющихся моделях лежат в `data/meta.json`.

Конфигурационные файлы имеющихся моделей лежат в директориях своих моделей: `models/{model_id}/config.yaml`