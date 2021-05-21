from csv import reader
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

x_training, y_training=[], []

def aux_func(cell):
    if cell=='': return_=0.0
    else: return_=float(cell)
    return return_

with open('SberMarket&1535_train.csv') as open_file:
    iterator=reader(open_file, delimiter=';')
    next(iterator)
    list_=list(iterator)
    for line in list_:
        x_training.append([aux_func(cell) for cell in line[1:-1]])
        y_training.append(float(line[-1]))

# Testing field size
tfs=int(input('Testing field size: '))
x_testing, y_testing=x_training[-tfs:], y_training[-tfs:]
x_training, y_training=x_training[:-tfs], y_training[:-tfs]

# Создаем линейную регрессию
model=ElasticNet(alpha=100)

# Обучаем модель на основе имеющихся данных
# Подбираем оптимальные значения весов (коэффициентов)
model.fit(x_training, y_training)

join_=lambda arg: '\n'.join(map(str, arg))

# Составляем предсказание количества заказов на основе тестового набора данных
y_predicted=model.predict(x_testing)
print('\nPrediction:\n'+join_(y_predicted))

# Реальные значения целевой переменной
print('\nTrue:\n'+join_(y_testing))

# Среднеквадратичная ошибка
print('\nMean squared error:', mean_squared_error(y_testing, y_predicted))

# Оптимальные значения весов
print('\nCoefficients:\n'+join_(model.coef_))