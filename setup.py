from setuptools import setup, find_packages

setup(
    name="md_prediction",
    version="0.1.0",
    description="Early prediction of multistate behavior in MD simulations",
    author="Ellie Lin, Ion Martinis, Vasanti Wall-Persad",
    author_email="persav@stanford.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "mdanalysis>=2.4.0",
        "deeptime>=0.4.3",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "temporal": [
            "keras-tcn",
            "mamba-ssm",
        ],
        "viz": [
            "matplotlib",
            "seaborn",
            "plotly",
        ],
    },
)