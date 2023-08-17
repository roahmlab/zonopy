from setuptools import setup
from setuptools import find_packages

setup(name='zonopy',
      version='0.0.1',
      install_requires=['torch','lxml','mat4py','gym'],
      packages=find_packages(
        include=['zonopy'],  # alternatively: `exclude=['additional*']`
      ),
      package_dir={"": "."}
)
