"""
Setup script for passive_walker_rl package
"""
from setuptools import setup, find_packages

# Read package info from egg-info
with open("passive_walker_rl.egg-info/top_level.txt", "r") as f:
    top_level = f.read().strip()

with open("passive_walker_rl.egg-info/entry_points.txt", "r") as f:
    entry_points = f.read()

# Parse entry points
entry_point_lines = entry_points.strip().split('\n')
console_scripts = []
for line in entry_point_lines:
    if line.startswith('[') or line.startswith('walker-'):
        if '=' in line:
            console_scripts.append(line)

setup(
    name="passive_walker_rl",
    version="2.1.0",
    description="Passive Walker RL Environment - Bipedal walking with FSM and neural network control",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "mujoco>=2.3.0",
        "gymnasium>=0.26.0",
        "torch>=1.9.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "pyyaml>=5.4.0",
        "matplotlib>=3.3.0",
        "pytest>=6.0.0",
    ],
    entry_points={
        "console_scripts": console_scripts,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
