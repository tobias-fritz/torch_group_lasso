import time
import torch
import matplotlib.pyplot as plt
from group_lasso import GroupLasso
from skglm import Lasso as SkglmGroupLasso

def benchmark_cuda_performance():
    """
    Benchmark GroupLasso.fit on varying dataset sizes and plot CPU vs CUDA performance
    along with skglm's L1GroupLasso.
    """
    device_cuda = torch.device("cuda") if torch.cuda.is_available() else None
    sample_sizes = [100, 500, 1000, 5000, 10000, 20000, 30000]   # different number of samples
    n_features = 50                     # fixed number of features
    cpu_times = []
    gpu_times = []
    skglm_times = []  # list to store skglm L1GroupLasso times
    groups = torch.arange(n_features, dtype=torch.int64)
    
    for n_samples in sample_sizes:
        # Generate synthetic data
        X = torch.randn(n_samples, n_features)
        beta_true = torch.randn(n_features)
        y = X @ beta_true + 0.1 * torch.randn(n_samples)
        
        # CPU benchmark for custom GroupLasso
        model_cpu = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=500, device='cpu')
        start = time.time()
        model_cpu.fit(X, y)
        cpu_times.append(time.time() - start)
        
        # GPU benchmark (if available) for custom GroupLasso
        if device_cuda is not None:
            X_cuda = X.to(device_cuda)
            y_cuda = y.to(device_cuda)
            model_gpu = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=500, device='cuda')
            start = time.time()
            model_gpu.fit(X_cuda, y_cuda)
            gpu_times.append(time.time() - start)
        else:
            gpu_times.append(None)

        # Benchmark for skglm's GroupLasso
        start = time.time()
        model_sk = SkglmGroupLasso(groups, fit_intercept=True, max_iter=500)  # skglm.GroupLasso expects groups as first positional arg
        model_sk.fit(X, y)
        skglm_times.append(time.time() - start)
        
        # clear memory
        del X, y, model_cpu, model_sk
        if device_cuda is not None:
            del X_cuda, y_cuda
        
    # Plot results
    plt.figure()
    plt.plot(sample_sizes, cpu_times, marker='o', label='CPU')
    if device_cuda is not None:
        plt.plot(sample_sizes, gpu_times, marker='o', label='GPU')
    plt.plot(sample_sizes, skglm_times, marker='o', label='skglm L1GroupLasso')  # new plot line
    plt.xlabel("Number of Samples")
    plt.ylabel("Time (seconds)")
    plt.title("GroupLasso.fit Time: CPU vs GPU vs skglm L1GroupLasso")
    plt.legend()
    plt.show()

# Uncomment below to run benchmark when executing this file directly.
# if __name__ == '__main__':
#     benchmark_cuda_performance()