from setuptools import setup

setup(
    name='cosserat_rod',
    version='0.0.1',
    description='Python implementation of numerical method for visco-elastic rods.',
    author='Tom Ranner, Tom Ilett', 'Lukas Deutz',
    url='https://github.com/LukasDeutz/cosserat_rod',
    packages=['cosserat_rod'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'fenics == 2019.1.0',
        'numpy >= 1.19, < 2',
        'scikit-learn >= 0.24'
    ],
    extras_require={
        'test': [
            'pytest'
        ],
        'inv': [
            'dolfin_adjoint @ git+https://github.com/dolfin-adjoint/pyadjoint.git@1c9c15c1fa2c1a470826143ce98b721ebd00facd',
            'torch >= 1.8, <= 1.9',
            'matplotlib >= 3.4',
            'tensorboard == 2.4.1',
        ]
    },
    python_requires=">=3.8, <3.10",
)
