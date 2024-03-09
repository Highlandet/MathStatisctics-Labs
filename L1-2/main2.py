import numpy as np
from utils import get_distribution, types, compute_trimmed_mean

def compute_stat_values():
    sizes = [10, 100, 1000]
    repeats = 1000

    for name in types:
        print(f"\nStatistics for {name} distribution:")
        print("{:<8} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "", "Mean", "Median", "z_r", "z_Q", "z_tr"
        ))

        for size in sizes:
            mean, med, zr, zq, ztr = [], [], [], [], []

            for _ in range(repeats):
                array = get_distribution(name, size)
                mean.append(np.mean(array))
                med.append(np.median(array))
                zr.append((max(array) + min(array)) / 2)
                zq.append((np.quantile(array, 0.25) + np.quantile(array, 0.75)) / 2)
                ztr.append(compute_trimmed_mean(array))

            print("{:<8}".format(f"n={size}"))

            print("{:<8} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                "E(z)", np.mean(mean), np.mean(med), np.mean(zr), np.mean(zq), np.mean(ztr)
            ))

            print("{:<8} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                "D(z)", np.mean(np.multiply(mean, mean)) - np.mean(mean) * np.mean(mean),
                np.mean(np.multiply(med, med)) - np.mean(med) * np.mean(med),
                np.mean(np.multiply(zr, zr)) - np.mean(zr) * np.mean(zr),
                np.mean(np.multiply(zq, zq)) - np.mean(zq) * np.mean(zq),
                np.mean(np.multiply(ztr, ztr)) - np.mean(ztr) * np.mean(ztr)
            ))


if __name__ == "__main__":
    compute_stat_values()


import numpy as np


def calculate_statistics(sample):
    # Выборочное среднее
    def Mean(sample):
        return sum(sample) / len(sample)

    # Выборочная медиана
    def Median(sample):
        if len(sample) % 2 == 1: return sample[len(sample) // 2]
        return 0.5 * (sample[len(sample) // 2] + sample[len(sample) // 2 + 1])

    # Полусумма экстремальных выборочных элементов
    def HalfsumExtreme(sample):
        return 0.5 * (sample[0] + sample[-1])

    # Полусумма квартилей
    def HalfsumQuartiles(sample):
        if len(sample) % 4 == 0:
            return 0.5 * (sample[len(sample) // 4 - 1] + sample[3 * len(sample) // 4 - 1]) / 2
        return 0.5 * (sample[len(sample) // 4] + sample[3 * len(sample) // 4]) / 2

    def TrimmedMean(sample):
        r = round(len(sample) / 4)
        return sum(sample[r:len(sample) - r + 1]) / 4

    return (Mean(sample), Median(sample), HalfsumExtreme(sample), HalfsumQuartiles(sample), TrimmedMean(sample))


def simulate_distribution(distribution, params, sample_size, iterations):
    for _ in range(sample_size):
        samples = None
        if distribution == 'normal':
            sample = np.random.normal(params[0], params[1], sample_size).tolist()
        elif distribution == 'cauchy':
            sample = np.random.standard_cauchy(sample_size).tolist()
        elif distribution == 'student':
            sample = np.random.standard_t(params[1], sample_size).tolist()
        elif distribution == 'poisson':
            sample = np.random.poisson(params[0], sample_size)
        elif distribution == 'uniform':
            sample = np.random.uniform(params[0], params[1], sample_size)
        else:
            raise ValueError("Invalid distribution specified.")
        return calculate_statistics(sample)


distributions = {
    'normal': (0, 1),
    'cauchy': (0, 1),
    'student': (0, 3),
    'poisson': (10,),
    'uniform': (-np.sqrt(3), np.sqrt(3))
}
sample_sizes = [10, 100, 1000]
iterations = 1000


def MeanResult(samples):
    transposed = [[row[i] for row in samples] for i in range(len(samples[0]))]
    return tuple([sum(transposed[i]) / len(transposed[i]) for i in range(len(transposed))])


def DispersionResult(samples):
    t_means = [[row[i] for row in samples] for i in range(len(samples[0]))]
    t_squares = [[row[i] ** 2 for row in samples] for i in range(len(samples[0]))]

    def MakerFunc(data):
        return sum(data[1]) / len(data[1]) - (sum(data[0]) / len(data[0])) ** 2

    return tuple([MakerFunc(tuple([t_means[i], t_squares[i]])) for i in range(len(t_means))])


for distribution, params in distributions.items():
    print(f"Statistics for {distribution.capitalize()} distribution:\n")
    for sample_size in sample_sizes:
        samples = []
        for i in range(iterations):
            samples.append(simulate_distribution(distribution, params, sample_size, iterations))
        mr, dr = MeanResult(samples), DispersionResult(samples)
        console_ans = f"""
    Size = {sample_size}:
    Means:
    {' & '.join(['{:.2f}'.format(m) for m in mr])} 
    Dispersion:
    {' & '.join(['{:.2f}'.format(d) for d in dr])}

    """
        print(console_ans)