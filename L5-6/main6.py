import numpy as np
from scipy.optimize import minimize

# Генерация данных
np.random.seed(42)  # Для воспроизводимости результатов
n = 20
x = np.linspace(-1.8, 2, n)
epsilons = np.random.normal(0, 1, n)
y = 2 + 2 * x + epsilons  # Эталонные значения

# Функция потерь для МНК по методу наименьших квадратов
def least_squares(params, y_t):
    a, b = params
    return np.sum((y_t - b - a * x)**2)

# Функция потерь для МНК по методу наименьших модулей
def least_absolute_deviation(params, y_t):
    a, b = params
    return np.sum(np.abs(y_t - b - a * x))

# Ищем МНК-оценки для метода наименьших квадратов
result_ls = minimize(least_squares, [0, 0], args=(y,))
a_ls, b_ls = result_ls.x

# Ищем МНК-оценки для метода наименьших модулей
result_lad = minimize(least_absolute_deviation, [0, 0], args=(y,))
a_lad, b_lad = result_lad.x

# Вывод результатов
print("Метод наименьших квадратов:")
print(f"a = {a_ls.round(3)}, b = {b_ls.round(3)}")
print("Метод наименьших модулей:")
print(f"a = {a_lad.round(3)}, b = {b_lad.round(3)}")

# Вносим возмущения
y_perturbed = y.copy()
y_perturbed[0] += 10
y_perturbed[-1] -= 10

# Ищем МНК-оценки для изменённой выборки методом наименьших квадратов
result_ls_perturbed = minimize(least_squares, [0, 0], args=(y_perturbed,))
a_ls_perturbed, b_ls_perturbed = result_ls_perturbed.x

# Ищем МНК-оценки для изменённой выборки методом наименьших модулей
result_lad_perturbed = minimize(least_absolute_deviation, [0, 0], args=(y_perturbed, ))
a_lad_perturbed, b_lad_perturbed = result_lad_perturbed.x

# Вывод результатов для изменённой выборки
print("\nИзменённая выборка:")
print("Метод наименьших квадратов:")
print(f"a = {a_ls_perturbed.round(3)}, b = {b_ls_perturbed.round(3)}")
print("Метод наименьших модулей:")
print(f"a = {a_lad_perturbed.round(3)}, b = {b_lad_perturbed.round(3)}")
