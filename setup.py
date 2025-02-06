# src/torch_group_lasso/__init__.py
# (empty file)

# setup.py
from setuptools import setup, find_packages

setup(
    name="torch_group_lasso",
    version="0.1.0",
    description="GroupLasso implemented with torch",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "skglm",
    ],
    entry_points={
        'console_scripts': [
            'benchmark-torch-group-lasso= torch_group_lasso.benchmark:benchmark_cuda_performance',
        ],
    },
)


