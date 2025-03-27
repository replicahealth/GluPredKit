from setuptools import setup, find_packages

project_name = "GluPredKit"
version = "1.0.19"
author = "Miriam K. Wolff"
author_email = "miriamkwolff@outlook.com"
package_name = "glupredkit"  # The package name on pip install

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=project_name,
    version=version,
    author=author,
    author_email=author_email,
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires='>=3.9',
    license='MIT',
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "reportlab",
        "svglib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "tidepool-data-science-project",
        "reportlab",
        "numpy",
        "aiohttp",
        "click",
        "dill",
        "python-nightscout",
        "error-grids",
        "methcomp",
        "wandb",
        "python-dotenv",
        "zipfile_deflate64",
    ],
    extras_require={
        'heavy': [
            "xport",
            "tensorflow==2.14.0",
            "keras-tcn",
            "torch",
            "opsb-pyloopkit==0.1.0",
            "py_replay_bg"
        ],
        'test': [
            "pytest",
            "pytest-ordering"
        ]
    },
    package_data={
        'glupredkit': ['unit_config.json'],
    },
    entry_points={
        'console_scripts': [
            'glupredkit = glupredkit.cli:cli',
        ],
    }
)
