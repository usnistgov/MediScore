#!/bin/bash


# This installation was tested on April 28, 2020 on Ubuntu 18.04.1 LTS
# This installation assumes that anaconda or miniconda is installed; conda is needed
# to install opencv 2.4, which is a critical package to get the current version to work
# This installation also requires sudo privlidges in order to apt-get install dependencies
# for the python packages, including version 1 packages of jasper.

sudo apt-get install -y libatlas-base-dev
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt-get update
sudo apt-get install -y libjasper1 libjasper-dev
sudo apt-get install -y libatlas3-base libwebp6 libtiff5 libjasper1 libilmbase12 libopenexr22 libilmbase12 libgstreamer1.0-0 libavcodec57 libavformat57 libavutil55 libswscale4 libqtgui4 libqt4-test libqtcore4
conda create -n mediscore python=2.7 # 2.7.17
conda activate mediscore
conda install -c conda-forge opencv=2.4
# opencv-python presumed to be installed through condo-forge
# These version numbers may matter
pip install numpy==1.15.4 pandas==0.21.1 matplotlib==2.0.2 scipy==1.2.1 scikit-learn==0.20.3 rawpy==0.10.1 numpngw==0.0.6 Glymur==0.8.12
# These version numbers do not matter
pip install bokeh jsonschema configparser
