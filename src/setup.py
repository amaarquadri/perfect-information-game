from setuptools import setup, find_packages

with open('README.md', 'r') as file:
    long_description = file.read()

setup(name='perfect-information-game',
      version='0.0.1',
      description='Create 2D perfect information board games, and play them with machine learning systems.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',
      py_modules=[],
      python_requires='>=3.8',
      install_requires=['numpy', ],
      extras_require={'dev': ['chess']},
      package_dir={'': 'src'},
      packages=find_packages(),
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.8',
                   'Operating System :: OS Independent'],
      url='https://github.com/amaarquadri/perfect-information-game',
      author='Amaar Quadri',
      author_email='amaarquadri@gmail.com')
# TODO: see if there are more relevant classifiers: https://pypi.org/classifiers/