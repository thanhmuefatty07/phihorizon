#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="supreme-system-v5",
    version="5.0.0",
    author="Supreme Trading Team",
    author_email="team@supreme-system.com",
    description="Real algorithmic trading system for ETH-USDT scalping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thanhmuefatty07/supreme-system-v5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "python-binance>=1.0.15",
        "ta>=0.10.1",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "supreme-system=cli:main",
        ],
    },
)
