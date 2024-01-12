from setuptools import setup, find_packages

setup(
    name="DataPlot",
    version="1.0",
    author="alex",
    author_email="alex870521@gmail.com",
    description="Aerosol science",

    url="",
    python_requires=">=3.6",

    # Specify your project's dependencies
    install_requires=[
        "pandas", "numpy", "matplotlib", "scipy"  # Add any other dependencies here
    ],
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)
