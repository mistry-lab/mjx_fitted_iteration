from setuptools import setup, find_packages

setup(
    name='diff_sim',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'diff_sim.xmls': ['*.xml'],
        'diff_sim.xmls.shadow_hand': ['*.xml', '*.png'],
        'diff_sim.xmls.shadow_hand.assets': ['*.*'],
    },
    install_requires=[
    ],
)
