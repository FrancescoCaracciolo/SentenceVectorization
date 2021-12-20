import setuptools

setuptools.setup(
    name="SentenceVectorization",
    version="0.0.1",
    url="https://github.com/FrancescoCaracciolo/SentenceVectorization",

    license="Apache Software License",

    author="Francesco Caracciolo",

    description="A library to rapresent sentences as vectors using GloVE",

    packages=[
        "SentenceVectorization",
    ],

    install_requires=[
        "numpy",
        "requests",
        "tqdm",
    ],

    include_package_data=True,
    zip_safe=False,

    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
)
