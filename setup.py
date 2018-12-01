from setuptools import setup, find_packages

setup(
      name='ann_inference' ,
      version='0.1dev' ,
      packages=find_packages() ,
      scripts=['boot.py'] ,
      long_description=open('README.txt').read()
      )
