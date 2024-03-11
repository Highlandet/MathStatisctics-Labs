import numpy as np
import matplotlib.pyplot as plt

sample_sizes = [20, 100]

distributions = {
    "Normal": (np.random.normal, {"loc": 0, "scale": 1}),
    "Cauchy": (np.random.standard_cauchy, {}),
    "Student-t": (np.random.standard_t, {"df": 3}),
    "Poisson": (np.random.poisson, {"lam": 10}),
    "Uniform": (np.random.uniform, {"low": -np.sqrt(3), "high": np.sqrt(3)})
}

for dist_name, (dist_func, dist_params) in distributions.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, sample_size in enumerate(sample_sizes):
        samples = dist_func(size=sample_size, **dist_params)
        axes[i].boxplot(samples, vert=False)
        axes[i].set_title(f"{dist_name}, n={sample_size}")
    plt.tight_layout()
    plt.show()
