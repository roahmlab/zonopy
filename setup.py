from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('zonopy/properties.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(name='zonopy',
      version=main_ns['__version__'],
      install_requires=['torch','numpy','scipy'],
      packages=find_packages(),
      package_dir={"": "."}
)
