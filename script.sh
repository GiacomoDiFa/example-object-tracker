#!/bin/sh

export DEMO_FILES="$HOME/demo_files"

cd /home/mendel/face_detection_GDF/example-object-tracker/gstreamer/

python3 detect.py   --videosrc /dev/video1   --videofmt raw   --tracker sort   --model ${DEMO_FILES}/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
