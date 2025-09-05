"""
Setup script for legacy passive walker package.
This allows the legacy code to be installed and used independently.
"""

from setuptools import setup, find_packages

setup(
    name="passive-walker-legacy",
    version="0.1.0",
    description="Legacy passive walker RL environment (frozen for reference)",
    long_description=open("README_LEGACY.md").read(),
    long_description_content_type="text/markdown",
    author="Yunus Emre Danabaş",
    author_email="yunusdanabas@su.edu.tr",
    packages=find_packages(),
    python_requires=">=3.9,<3.11",
    install_requires=[
        "jax",
        "equinox",
        "optax",
        "brax",
        "gym",
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "mujoco",
        "pyyaml",
    ],
    classifiers=[
        "Development Status :: 7 - Inactive",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="reinforcement-learning mujoco bipedal-walker legacy",
)
