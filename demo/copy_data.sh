#!/bin/bash

REGRESSION_PATH=$PWD/../regression
DEMO_PATH=$PWD/./

# copy model and dataset.
cp ${REGRESSION_PATH}/model/yolov5s.onnx $DEMO_PATH
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 $DEMO_PATH
cp -rf ${REGRESSION_PATH}/image $DEMO_PATH
