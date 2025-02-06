import time
import torch
import matplotlib.pyplot as plt
from group_lasso import GroupLasso

def benchmark_cuda_performance():
    """
    Benchmark GroupLasso.fit on varying dataset sizes and plot CPU vs CUDA performance.
    """
    device_cuda = torch.device("cuda") if torch.cuda.is_available() else None
    sample_sizes = [100, 500, 1000, 5000, 10000]  # different number of samples
    n_features = 50                     # fixed number of features
    cpu_times = []
    gpu_times = []
    groups = torch.arange(n_features, dtype=torch.int64)
    
    for n_samples in sample_sizes:
        # Generate synthetic data
        X = torch.randn(n_samples, n_features)
        beta_true = torch.randn(n_features)
        y = X @ beta_true + 0.1 * torch.randn(n_samples)
        
        # CPU benchmark
        model_cpu = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=100, device='cpu')
        start = time.time()
        model_cpu.fit(X, y)
        cpu_times.append(time.time() - start)
        
        # GPU benchmark (if available)
        if device_cuda is not None:
            X_cuda = X.to(device_cuda)
            y_cuda = y.to(device_cuda)
            model_gpu = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=100, device='cuda')
            start = time.time()
            model_gpu.fit(X_cuda, y_cuda)
            gpu_times.append(time.time() - start)
        else:
            gpu_times.append(None)
        # clear memory
        del X, y, model_cpu, model_gpu
    
    # Plot results
    plt.figure()
    plt.plot(sample_sizes, cpu_times, marker='o', label='CPU')
    if device_cuda is not None:
        plt.plot(sample_sizes, gpu_times, marker='o', label='GPU')
    plt.xlabel("Number of Samples")
    plt.ylabel("Time (seconds)")
    plt.title("GroupLasso.fit Time: CPU vs GPU")
    plt.legend()
    plt.show()

# Uncomment below to run benchmark when executing this file directly.
# if __name__ == '__main__':
#     benchmark_cuda_performance()