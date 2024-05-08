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
        "pandas",
        "numpy",
        "torch",
        "scipy",
        "pyyaml",
        "h5py",
        "einops",
        "tensorboard",
        'pyteomics',
        'tqdm'

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    version="0.0.1",
)
