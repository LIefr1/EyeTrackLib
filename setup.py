from setuptools import find_packages, setup

setup(
    name="EyeTrackLib",
    version="0.1",
    description="Library for webcam eye tracking",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    zip_safe=False,
)
