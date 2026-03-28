#!/bin/bash
# Download and setup Cityscapes on TPU VM
# Usage: ./download_cityscapes_tpu.sh <username> <password>

USERNAME=$1
PASSWORD=$2

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Usage: $0 <username> <password>"
    exit 1
fi

export DATASET_DIR=~/datasets/cityscapes
mkdir -p $DATASET_DIR
cd $DATASET_DIR

echo "Logging into Cityscapes..."
wget --keep-session-cookies --save-cookies=cookies.txt \
     --post-data "username=${USERNAME}&password=${PASSWORD}&submit=Login" \
     https://www.cityscapes-dataset.com/login/

echo "Downloading standard image sets and labels..."

# Download leftImg8bit_trainvaltest.zip (ID 3)
wget --load-cookies cookies.txt --content-disposition "https://www.cityscapes-dataset.com/file-handling/?packageID=3"

# Download rightImg8bit_trainvaltest.zip (ID 4)
wget --load-cookies cookies.txt --content-disposition "https://www.cityscapes-dataset.com/file-handling/?packageID=4"

# Download gtFine_trainvaltest.zip (ID 1)
wget --load-cookies cookies.txt --content-disposition "https://www.cityscapes-dataset.com/file-handling/?packageID=1"

# Download camera_trainvaltest.zip (ID 8)
wget --load-cookies cookies.txt --content-disposition "https://www.cityscapes-dataset.com/file-handling/?packageID=8"

echo "Extracting datasets..."
unzip -q leftImg8bit_trainvaltest.zip
unzip -q rightImg8bit_trainvaltest.zip
unzip -q gtFine_trainvaltest.zip
unzip -q camera_trainvaltest.zip

echo "Cleanup..."
rm cookies.txt *.zip

echo "Cityscapes setup complete in $DATASET_DIR"
ls -d */
