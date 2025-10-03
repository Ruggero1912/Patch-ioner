"""
Setup script for AlphaCLIP Standalone
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README if it exists
readme_content = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()

setup(
    name="alphaclip-standalone",
    version="1.0.0",
    author="AlphaCLIP Team",
    description="Standalone version of AlphaCLIP for easy integration",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        'alpha_clip': ['*.gz'],  # Include the tokenizer vocabulary file
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="clip, vision, language, deep learning, pytorch",
)
