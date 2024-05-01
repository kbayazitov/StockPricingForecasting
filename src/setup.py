import io
import re
from setuptools import setup, find_packages

from mylib import __version__

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')
# вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    # metadata
    name='mylib',
    version=__version__,
    license='MIT',
    author='Kamil Bayazitov',
    author_email="baiazitov.km@phystech.edu",
    description='mylib, python package',
    long_description=readme,
    url='https://github.com/kbayazitov/StockPricingForecasting',

    # options
    packages=find_packages(),
    install_requires=requirements,
)
