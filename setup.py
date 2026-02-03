from setuptools import setup, find_packages
setup(
    name="nuble",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "rich",
        "requests",
        "inquirer",
        "tiktoken",
        "openai",
        "anthropic",
        "python-dotenv",
        "numpy",
    ],
    entry_points={
        'console_scripts': [
            'nuble=nuble.cli:main',
        ],
    },
    python_requires=">=3.8",
)