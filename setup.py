from setuptools import setup, find_packages, os

VERSION = '0.0.1'
DESCRIPTION = 'A simple SLAM implementation in python for learning'
LONG_DESCRIPTION = 'A simple SLAM(Simultaneous Localization and Mapping) implementation in python for learning'

# Read the contents of your requirements.txt file

def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path) as requirements_file:
        return requirements_file.read().splitlines()

# Setting up
setup(
    name="spslam",
    version=VERSION,
    author="Ali Kuwajerwala <alihkw.com>",
    author_email="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=read_requirements(),
    keywords=['python', 'SLAM', 'Simultaneous Localization and Mapping'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Ubuntu",
    ]
)