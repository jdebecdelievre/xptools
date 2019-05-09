from setuptools import setup, find_packages


def get_long_description():
    with open("README.md", "r") as readme_file:
        return readme_file.read()


setup(
    name='xptools',
    description='Tools for experimental data treatment',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    version='0.0.1',
    author='jeandb',
    author_email='jeandb@stanford.edu',
    license='MIT',)
