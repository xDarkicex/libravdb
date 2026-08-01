from setuptools import setup, find_packages

setup(
    name="libravdb",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "libravdb": ["ext/*"],
    },
    # Ensure it's not installed as a zipped egg so the .so can be loaded
    zip_safe=False,
)
