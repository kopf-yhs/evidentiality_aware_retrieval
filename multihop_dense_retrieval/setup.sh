#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

conda install pytorch==1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install faiss-gpu pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

python setup.py develop