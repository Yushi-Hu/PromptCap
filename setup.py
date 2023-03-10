from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='promptcap',
    packages=['promptcap'],
    version='1.0.3',
    license='MIT',
    description='Instruction-Guided Image Captioning',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Yushi Hu',
    author_email='yushihu@uw.edu',
    url='https://github.com/yushihu/PromptCap',
    keywords=['image captioning', 'image-to-text','instruction', 'natural language processing', 'computer vision'],
)
