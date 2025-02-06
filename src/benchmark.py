import time
import torch
import matplotlib.pyplot as plt
import numpy as np             # new import
from group_lasso import GroupLasso
from skglm import Lasso as SkglmGroupLasso
import gc

def benchmark_cuda_performance():
    """
    Benchmark GroupLasso.fit on varying dataset sizes and plot CPU vs CUDA performance
    along with skglm's L1GroupLasso.
    """
    device_cuda = torch.device("cuda") if torch.cuda.is_available() else None
    sample_sizes = [100, 500, 1000, 5000, 10000, 20000, 30000]   # different number of samples
    n_runs = 5                                # number of runs per sample size
    n_features = 50                           # fixed number of features
    cpu_times_mean, cpu_times_std = [], []
    gpu_times_mean, gpu_times_std = [], []
    skglm_times_mean, skglm_times_std = [], []
    groups = torch.arange(n_features, dtype=torch.int64)
    
    for n_samples in sample_sizes:
        cpu_runs, gpu_runs, sk_runs = [], [], []
        for _ in range(n_runs):
            # Generate synthetic data
            X = torch.randn(n_samples, n_features)
            beta_true = torch.randn(n_features)
            y = X @ beta_true + 0.1 * torch.randn(n_samples)
            
            # CPU benchmark for custom GroupLasso
            model_cpu = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=500, device='cpu')
            start = time.time()
            model_cpu.fit(X, y)
            cpu_runs.append(time.time() - start)
            
            # GPU benchmark (if available) for custom GroupLasso
            if device_cuda is not None:
                X_cuda = X.to(device_cuda)
                y_cuda = y.to(device_cuda)
                model_gpu = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=500, device='cuda')
                start = time.time()
                model_gpu.fit(X_cuda, y_cuda)
                gpu_runs.append(time.time() - start)
                del X_cuda, y_cuda
            else:
                gpu_runs.append(None)
            
            # Benchmark for skglm's GroupLasso
            start = time.time()
            model_sk = SkglmGroupLasso(groups.detach().numpy(), fit_intercept=True, max_iter=100)
            model_sk.fit(X.detach().numpy(), y.detach().numpy())
            sk_runs.append(time.time() - start)
            
            # clear memory
            del X, y, model_cpu, model_sk
            gc.collect()
            if device_cuda is not None:
                torch.cuda.empty_cache()
        # Compute statistics
        cpu_times_mean.append(np.mean(cpu_runs))
        cpu_times_std.append(np.std(cpu_runs))
        if device_cuda is not None:
            gpu_times_mean.append(np.mean(gpu_runs))
            gpu_times_std.append(np.std(gpu_runs))
        else:
            gpu_times_mean.append(None)
            gpu_times_std.append(None)
        skglm_times_mean.append(np.mean(sk_runs))
        skglm_times_std.append(np.std(sk_runs))
    
    # Plot results with error bars
    plt.figure()
    plt.errorbar(sample_sizes, cpu_times_mean, yerr=cpu_times_std, marker='o', label='CPU')
    if device_cuda is not None:
        plt.errorbar(sample_sizes, gpu_times_mean, yerr=gpu_times_std, marker='o', label='GPU')
    plt.errorbar(sample_sizes, skglm_times_mean, yerr=skglm_times_std, marker='o', label='skglm L1GroupLasso')
    plt.xlabel("Number of Samples")
    plt.ylabel("Time (seconds)")
    plt.title("GroupLasso.fit Time: CPU vs GPU vs skglm L1GroupLasso")
    plt.legend()
    plt.show()

# Uncomment below to run benchmark when executing this file directly.
# if __name__ == '__main__':
#     benchmark_cuda_performance()