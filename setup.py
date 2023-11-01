from setuptools import setup, find_packages

setup(name='zonopy',
      version='0.1.0',
      install_requires=['torch','numpy'],
      packages=find_packages(),
      package_dir={"": "."}
)
