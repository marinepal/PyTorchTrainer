import setuptools

with open("requirements.txt", "r") as requirements_file:
    reqs = requirements_file.read().splitlines()

setuptools.setup(
    name='PyTorchTrainer',
    version=0.0001,
    package_dir={'': 'src'},
    packages=setuptools.find_namespace_packages(where='src'),
    description="Trainer with 3 models on CIFAR10 dataset",
    long_description=open('README.md').read(),
    install_requires=reqs,
    python_requires=">=3.6"
)
