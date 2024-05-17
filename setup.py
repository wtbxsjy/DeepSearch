from setuptools import setup, find_packages

with open("README.md", "r") as source:
    long_description = source.read()


setup(
    name="DeepSearch",
    author="Y. Y",
    packages=find_packages(),
    include_package_data=True,
    description="Contrastive tandem mass spectrometry database search engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas==1.5.3",
        "numpy==1.24.1",
        "torch==2.0.1",
        "scipy==1.10.1",
        "pyyaml==6.0",
        "h5py==3.8.0",
        "einops==0.6.1",
        "tensorboard==2.12.0",
        'pyteomics==4.5.6',
        'tqdm==4.65.0'

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    version="1.0",
)
