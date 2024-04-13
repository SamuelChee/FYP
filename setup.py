import setuptools


setuptools.setup(
    name="opencv_optical_flow",
    version="1.0",
    packages=["opencv_optical_flow", "robotcar_dataset_sdk"],
    package_dir={
        "opencv_optical_flow": "opencv_optical_flow",
        "robotcar_dataset_sdk": "robotcar-dataset-sdk/python",
    },
    entry_points={
        "console_scripts": [
            "radar_cv_flow = opencv_optical_flow.radar_cv_flow:main",
        ],
    },
    # Other configuration options...
)
