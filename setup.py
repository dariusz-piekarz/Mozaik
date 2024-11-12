from setuptools import setup, find_packages


setup(
    name="mozaik",
    version="0.1",
    author="Dariusz Piekarz",
    author_email="darpiekarz@wp.pl",
    description="Provided main image and directory with other images it replace the original picture pixels' with"
                "the other images and glue them together.",

    url="",
    packages=find_packages(),
    install_requires=[
        'numpy=1.26.1',
        'pillow=10.4.0',
        'loguru=0.7.2',
        'joblib=1.4.2',
        'pytools=0.0.1'
    ],
    package_data={'': ['config.json']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
           'mozaik=main:main',
           'mozaik2=main:main2'
        ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.12.0",
)
