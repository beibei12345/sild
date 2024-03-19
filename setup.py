from setuptools import setup, find_packages

__version__ = '0.0.1'
URL = None
install_requires = [
    "matplotlib",
    "libwon",
]

setup(
    name='sild',
    version=__version__,
    description='sild',
    author='mnlab',
    url=URL,
    python_requires='>=3.9',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
