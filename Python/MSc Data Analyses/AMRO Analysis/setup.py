from setuptools import setup, find_packages

setup(
    name="amro",
    version="0.1.0",
    author="James Fraser",
    author_email="james.fraser@umontreal.ca",
    description="Angle-resolved Magnetoresistance Oscillation (AMRO) Analysis by Fourier transform and best fit.",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "lmfit>=1.2.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.9",
)
