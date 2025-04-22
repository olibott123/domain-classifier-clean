from setuptools import setup, find_packages

setup(
    name="domain_classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "snowflake-connector-python>=3.0.0",
        "numpy>=1.16.0",
        "requests>=2.0.0",
        "gunicorn>=20.0.0",
        "flask-cors>=4.0.0",
        "python-dotenv>=1.0.0",
        "beautifulsoup4>=4.11.1"
    ],
)
