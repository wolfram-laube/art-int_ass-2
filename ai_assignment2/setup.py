from setuptools import setup, find_packages

setup(
    name='ai_assignments',
    version='0.1',
    url='https://gitlab.cp.jku.at/artificial_intelligence/problem_instance_generator',
    author='Rainer Kelz',
    author_email='rainer.kelz@jku.at',
    description='Assignment Framework for AI Excercise',
    packages=find_packages(),
    install_requires=[
        'lxml >= 4.5.2',
        'numpy >= 1.19.2',
        'matplotlib >= 3.3.2',
        'networkx >= 2.5',
        'pydot >= 1.4.1',
        'scikit-learn >= 0.23.0',
        'graphviz >= 0.19'
    ],
)
