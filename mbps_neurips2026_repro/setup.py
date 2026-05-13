"""Setup script for MBPS - Mamba-Bridge Panoptic Segmentation."""
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mbps",
    version="0.1.0",
    description="Unsupervised Mamba-Bridge Panoptic Segmentation",
    author="MBPS Research Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mbps-train=scripts.train:main",
            "mbps-eval=scripts.evaluate:main",
        ],
    },
)



