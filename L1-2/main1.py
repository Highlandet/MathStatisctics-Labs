# Задание 1

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

bins_table = {10 : 8, 50: 16, 1000: 32}

# Заданные распределения
distributions = [
    ("Normal", lambda size: np.random.normal(0, 1, size)),
    ("Cauchy", lambda size: np.random.standard_cauchy(size)),
    ("Student", lambda size: np.random.standard_t(3, size)),
    ("Poisson", lambda size: np.random.poisson(10, size)),
    ("Uniform", lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size))
]

# Размеры выборок
sample_sizes = [10, 50, 1000]

for dist_name, dist_func in distributions:
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(dist_name)
    for i, size in enumerate(sample_sizes):
        # Генерация выборки
        sample = dist_func(size)
        # Определение расположения подграфиков
        ax = plt.subplot2grid((3, 2), (i // 2, i % 2))
        # Построение гистограммы
        ax.hist(sample, bins=bins_table[size], density=True, alpha=0.5, label='Histogram')

        # Построение графика плотности распределения
        x = np.linspace(min(sample), max(sample), 1000)
        if dist_name == "Uniform":
            y = stats.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3))
        else:
            y = stats.gaussian_kde(sample).evaluate(x)
        ax.plot(x, y, color='red', label='Density')

        ax.set_title(f"n={size}")
        ax.legend()

    plt.tight_layout()
    plt.show()

