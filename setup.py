import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "TI-Lib",
    version = "0.0.1",
    author = "Nitul Rupareliya",
    author_email = "nvr.job99@gmail.com",
    description = "This is a technical indicators library which is highly useful in the technical analysis of stock market data.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Nitul067/TI-Lib",
    classifiers = [
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    install_requires = [
          "numpy",
          "pandas" 
    ],
    python_requires = ">=3.10"
)