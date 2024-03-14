import setuptools

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="python-voi",
    version="0.0.2",
    author="Core Francisco Park",
    author_email="cfpark00@gmail.com",
    description="Python implementation of Variation of Information",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    install_requires=["numpy","torch"],
)
