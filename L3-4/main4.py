import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

"""
Таблица обозначений:
---------------------+-------------------
    mean (Mu)        | выборочное среднее
---------------------+-------------------
    std (Sigma)      | выборочное СКО
---------------------+-------------------
    NthCentralMoment | n-й центральный
                     | момент
---------------------+-------------------
    excess           | эксцесс
---------------------+-------------------
    gothic_u         | смещение СКО 
                     | (относительное)   
---------------------+-------------------
"""
#   Размеры выборок
sample_sizes = [20, 100]

#   Регуляция столбцов гистограммы
bins_table = {20: 10, 100: 32}

#   Распределения
distributions = {
    "Normal": (np.random.normal, {"loc": 0, "scale": 1}),
    "Random": (np.random.random, {}),
}

#   Функции поиска квантилей (распределения Стьюдента, Гаусса и Пирсона)
qntl_t = sts.t.ppf #    Стьюдент
qntl_chi2 = sts.chi2.ppf #    Пирсон
qntl_nl = sts.norm.ppf #    Гаусс



class Sample:

    #   Инициализация выборки
    def __init__(self, dist_func, sample_size, **dist_params):
        self.sample = dist_func(size=sample_size, **dist_params)

    #   Поиск границ ДИ среднего у ГНС
    def NormalMuBounds(self, sgnfnce):
        sample_size = len(self.sample)
        mean, std = np.mean(self.sample), np.std(self.sample)
        quantile = qntl_t(1 - sgnfnce / 2, sample_size - 1)
        MuOffset = std * quantile / np.sqrt(sample_size - 1)
        return mean - MuOffset, mean + MuOffset

    #   Поиск границ ДИ СКО у ГНС
    def NormalSigmaBounds(self, sgnfnce):
        sample_size = len(self.sample)
        mean, std = np.mean(self.sample), np.std(self.sample)  # Выборочные среднее и дисперсия
        SgmNum = std * np.sqrt(sample_size)
        return SgmNum / np.sqrt(qntl_chi2(1-sgnfnce / 2, sample_size - 1)), \
            SgmNum / np.sqrt(qntl_chi2(sgnfnce / 2, sample_size - 1))

    #   Поиск границ ДИ среднего у произвольной выборки
    def RandomMuBounds(self, sgnfnce):
        sample_size = len(self.sample)
        mean, std = np.mean(self.sample), np.std(self.sample)  # Выборочные среднее и дисперсия
        quantile = qntl_nl(1-sgnfnce/2)
        MuOffset = std * quantile / np.sqrt(sample_size - 1)
        return mean - MuOffset, mean + MuOffset

    #   Поиск границ ДИ СКО у произвольной выборки
    def RandomSigmaBounds(self, sgnfnce):
        std, sample_size = np.std(self.sample), len(self.sample)
        NthCentralMonent = lambda N: np.sum((self.sample - np.mean(self.sample))**N) / sample_size
        excess = NthCentralMonent(4) / std**4 - 3
        gothic_u = qntl_nl(1-sgnfnce/2) * np.sqrt((excess + 2) / sample_size)
        return std * (1-0.5*gothic_u), std * (1+0.5*gothic_u)


sgms_normal, sgms_random = [], []

def Informing(sample_size):
    sgnfnce = float(input("Enter the signifance level (choose from <0, 1>-interval):"))
    my_sample = Sample(dist_func, sample_size, **dist_params)  # Выборка
    if dist_name == "Random": my_sample.sample = (my_sample.sample - np.min(my_sample.sample)) / (np.max(my_sample.sample))

    if dist_name == "Normal":
        mu_min, mu_max = my_sample.NormalMuBounds(sgnfnce)  # Границы доверительного интервала для среднего для ГНС
        sgm_min, sgm_max = my_sample.NormalSigmaBounds(sgnfnce)  # Границы доверительного интервала для СКО для ГНС
        sgms_normal.append((sgm_min, sgm_max))
    if dist_name == "Random":
        mu_min, mu_max = my_sample.RandomMuBounds(sgnfnce)  # Границы доверительного интервала для среднего для произвольный выборки
        sgm_min, sgm_max = my_sample.RandomSigmaBounds(sgnfnce)  # Границы доверительного интервала для СКО для произвольной выборки
        sgms_random.append((sgm_min, sgm_max))

    print(f"""{dist_name} Distribution: size = {sample_size}
    {mu_min} < Mu < {mu_max}
    {sgm_min} < Sigma < {sgm_max}\n 
""")
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"{dist_name}. N = {sample_size}")
    labels = [
        'Histogram', r'$\mu_{\min}$', r'$\mu_{\max}$',
        r'$\mu_{\min} - \sigma_{\max}$', r'$\mu_{\max} + \sigma_{\max}$'
    ]
    plt.hist(my_sample.sample, bins=bins_table[sample_size], density=True, alpha=0.5, label='Histogram')
    color_table = {0: 'r', 1: 'b'}
    params = [mu_min, mu_max, mu_min - sgm_max, mu_max + sgm_max]
    for i, param in enumerate(params):
        plt.axvline(param, color=color_table[i // 2], linestyle='-', linewidth=1, label='Mean',marker='o')
    plt.show()


#   Построение доверительных интервалов СКО
def PlotSSO(sgms):
    plt.hlines(0.5, xmin=sgms[0][0], xmax=sgms[0][1], color='b', linestyles='-')
    plt.hlines(0.7, xmin=sgms[1][0], xmax=sgms[1][1], color='r', linestyles='-')
    plt.plot([sgms[0][0], sgms[0][1]], [0.5, 0.5], 'ro', markersize=5)
    plt.plot([sgms[1][0], sgms[1][1]], [0.7, 0.7], 'ro', markersize=5)
    plt.legend(labels=[r'$N=20: [\sigma_{\min}; \sigma_{\max}]$',
                       r'$N=100: [\sigma_{\min}; \sigma_{\max}]$'
                       ])
    plt.show()

for dist_name, (dist_func, dist_params) in distributions.items():
    for i, sample_size in enumerate(sample_sizes):
        Informing(sample_size)
PlotSSO(sgms_normal)
PlotSSO(sgms_random)