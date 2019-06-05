#!/bin/bash

for f in /Testing/*.mp4
do
	python main.py -v $f -l -r ${f::-4}_hungarian.avi
done


