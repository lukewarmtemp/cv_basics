from setuptools import setup

package_name = 'cv_basics'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='luke.jia.yang@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'camera = cv_basics.test_camera:main',
        'segment = cv_basics.yolo_seg_node:main',
        'main = cv_basics.main_run:main'
        ],
    },
)
