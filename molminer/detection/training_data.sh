#!/bin/bash

YOLO_DATA_CHEM="yolo_clef_data"

if [ -d ./datasets/data/${YOLO_DATA_CHEM} ]
then
	echo "CLEF-2012 IP data already downloaded."
else
  rm -f yolo_clef_data.zip
  mkdir -p ./datasets/data
	wget https://www.cs.rit.edu/~dprl/data/${YOLO_DATA_CHEM}.zip && unzip ${YOLO_DATA_CHEM}.zip -d ./datasets/data
fi