from setuptools import setup, find_packages

setup(
    name="DataPlot",
    version="1.0",
    author="alex",
    author_email="alex870521@gmail.com",
    description="Aerosol science",

    url="",
    python_requires=">=3.12",

    # Specify your project's dependencies
    install_requires=[
        "pandas", "numpy", "matplotlib", "scipy", "seaborn", "scikit-learn", "windrose",
        "PyMieScatt",
        "tabulate", "tqdm"
        # Add any other dependencies here
    ],
    packages=find_packages()
)
