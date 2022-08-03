from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description="Creates a Machine Learning and Natural Language Processing based model, and serves it up via FastAPI. The API takes 2 questions as input, and returns it's similarity score",
    author='Aditya Jha',
    license='MIT',
)
