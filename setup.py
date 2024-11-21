# setup.py
from setuptools import setup, find_packages

setup(
    name="clustering_classes_pca",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'mrcfile',
        'scikit-image',
    ],
    entry_points={
        'console_scripts': [
            'clustering_images_pca=clustering_images_pca:main',
        ],
    },
)