#!/usr/bin/env bash

# Step 1: Install numpy and cython first
pip install numpy==1.24.3 cython>=0.29

# Step 2: Install the rest of the requirements
pip install -r requirements.txt
