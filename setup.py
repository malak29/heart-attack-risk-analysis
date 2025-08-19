from setuptools import setup, find_packages

setup(
    name="heart-attack-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Production ML system for heart attack prediction",
)