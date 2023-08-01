from setuptools import setup
from glob import glob
import os

package_name = 'F1TenthRacingROS'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('maps/*.csv')),
        # (os.path.join('share', package_name), glob('F1TenthRacingROS/Utils/*.csv')),
        # (os.path.join('share', package_name), glob('maps/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benjy',
    maintainer_email='benjaminevans316@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit=F1TenthRacingROS.PurePursuitNode:main',
            'nn_agent=F1TenthRacingROS.AgentNode:main',
            'pp_agent=F1TenthRacingROS.PurePursuit:main',
        ],
    },
)
