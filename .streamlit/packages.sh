#!/bin/bash
echo "Running packages.sh..."
apt-get update
apt-get install -y libgl1 libgl1-mesa-glx
echo "libGL packages installed."
