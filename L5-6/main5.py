import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from matplotlib.patches import Ellipse

# Функция для генерации выборок
def generate_samples(size, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, size)

# Функция для вычисления коэффициента квадрантной корреляции
def quadrant_correlation(x, y):
    return np.mean(np.sign(x) == np.sign(y))

# Средний квадрат
def squares_mean(data):
    return np.mean(np.array(data) ** 2)

# Дисперсия
def variance(data):
    return np.mean((data - np.mean(data)) ** 2)

# Параметры задания
sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
num_samples = 1000

#Визуализатор выборки и эллипса равновероятности
def visualize(samples, name):
    plt.figure(figsize=(12, 8))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)

    #Отрисовка эллипса
    def ellipse_maker(ax, cov, mean, n_std=2, **kwargs):
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = Ellipse(mean, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellipse)

    ellipse_maker(plt.gca(), np.cov(samples.T), np.mean(samples, axis=0), edgecolor='r', linestyle='--', fill=False)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'EP-Ellipse of {name}')
    plt.grid(True)
    plt.show()

# Вывод информации
def inform(data, coeff_name):
    print(f"""
    {coeff_name}:
        Mean: {np.mean(data).round(3)}
        Squares Mean: {squares_mean(data).round(3)}
        Variance: {variance(data).round(3)}
""")

# Вычисление результатов для НДР
for size in sizes:
    for rho in rhos:
        print(f"Size: {size}, Rho: {rho}")
        pearson_corrs, spearman_corrs, quadrant_corrs = [], [], []

        for _ in range(num_samples):
            samples = generate_samples(size, rho)
            x = samples[:, 0]
            y = samples[:, 1]

            pearson_corr, _nan_ = pearsonr(x, y)
            spearman_corr, _nan_ = spearmanr(x, y)
            quadrant_corr = quadrant_correlation(x, y)
            del _nan_

            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            quadrant_corrs.append(quadrant_corr)

        inform(pearson_corrs, "Pearson")
        inform(quadrant_corrs, "Quadrant")
        inform(spearman_corrs, "Spearman")

# Генерация смешанной выборки
def generate_mixed(size):
    return 0.9 * generate_samples(size, 0.9) + 0.1 * generate_samples(size, -0.9)

# Вычисления смешанной выборки
for size_ in sizes:
    print(f"Mixed Distribution\nSize: {size_}\n")
    pearson_corrs, spearman_corrs, quadrant_corrs = [], [], []
    for _ in range(num_samples):
        mix_samples = generate_mixed(size_)
        x = mix_samples[:, 0]
        y = mix_samples[:, 1]

        pearson_corr, _nan_ = pearsonr(x, y)
        spearman_corr, _nan_ = spearmanr(x, y)
        quadrant_corr = quadrant_correlation(x, y)
        del _nan_

        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        quadrant_corrs.append(quadrant_corr)

    inform(pearson_corrs, "Pearson")
    inform(quadrant_corrs, "Quadrant")
    inform(spearman_corrs, "Spearman")

# Построение графиков
for size_ in sizes:
    visualize(generate_mixed(size_), f"mixed distribution, N={size_}")