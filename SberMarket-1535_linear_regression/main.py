from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

data = read_csv('data.csv')
data = data.fillna(0)

tfs = int(input('Testing field size: '))

x_training = data.iloc[:-tfs, 1:-1]
y_training = data.iloc[:-tfs, -1]

x_testing = data.iloc[-tfs:, 1:-1]
y_testing = data.iloc[-tfs:, -1]

weeks = data['week'][-tfs:]

# Создаем линейную регрессию
model = ElasticNet(alpha=100)

# Обучаем модель на основе имеющихся данных
# Подбираем оптимальные значения весов (коэффициентов)
model.fit(x_training, y_training)

join_ = lambda arg: '\n'.join(map(str, arg))

# Составляем предсказание количества заказов на основе тестового набора данных
y_predicted = model.predict(x_testing)
print('\nPrediction:\n' + join_(y_predicted))

# Реальные значения целевой переменной
print('\nTrue:\n' + join_(y_testing))

# Среднеквадратичная ошибка
print('\nMean squared error:', mean_squared_error(y_testing, y_predicted))

# Оптимальные значения весов
print('\nCoefficients:\n' + join_(model.coef_))

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