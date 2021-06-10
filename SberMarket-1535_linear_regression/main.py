from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data = read_csv('data.csv')
# Сортируем по дате
data = data.sort_values('week')
# Заменяем все пропуски на 0
data = data.fillna(0)

tfs = int(input('Testing field size: '))

weeks = data['week'][-tfs:]
# Исключаем колонки 'week', 'app_clicks'
data = data.drop(columns=['week', 'app_clicks'])
feature_names = list(data.head(1))

# Обучающая и тестовая выборки
x_training = data.iloc[:-tfs, :-1]
y_training = data.iloc[:-tfs, -1]

x_testing = data.iloc[-tfs:, :-1]
y_testing = data.iloc[-tfs:, -1]

# Подбор оптимального значения константы alpha
'''
mse_ = 100000000000
alpha_ = 0

for i in range(10000, 20000):
    model = ElasticNet(alpha=i)
    model.fit(x_training, y_training)
    y_predicted = model.predict(x_testing)
    mse = mean_squared_error(y_testing, y_predicted)
    if mse < mse_:
        mse_ = mse
        alpha_ = i

print(alpha_)
'''

# Создаем линейную регрессию
model = ElasticNet(alpha=14898)

# Обучаем модель на основе имеющихся данных
# Подбираем оптимальные значения весов (коэффициентов)
model.fit(x_training, y_training)

join_func = lambda arg: '\n'.join(map(str, arg))

# Составляем предсказание количества заказов на основе тестового набора данных
y_predicted = model.predict(x_testing)
print('\nPrediction:\n' + join_func(y_predicted))

# Реальные значения целевой переменной
print('\nTrue:\n' + join_func(y_testing))

# Среднеквадратичная ошибка
print('\nMean squared error:', mean_squared_error(y_testing, y_predicted))

# Средняя абсолютная ошибка в процентах
print('\nMean absolute percentage error:', mean_absolute_percentage_error(y_testing, y_predicted))

# Оптимальные значения весов
print('\nCoefficients:')
for i in range(len(model.coef_)):
    feature = feature_names[i]
    print(feature, '-'*(30-len(feature)), model.coef_[i])

# Визуализация
manager = plt.get_current_fig_manager()
manager.canvas.set_window_title('')

plt.xlabel('week')
plt.ylabel('order_app')

plt.plot(weeks, y_testing, 'o', label='y_testing')
plt.plot(weeks, y_predicted, 'ro', label='y_predicted')

plt.grid()
plt.legend()

plt.show()
