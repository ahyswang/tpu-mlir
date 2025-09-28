#!/bin/bash

# ../../build/bin/tpuc-opt yolov5s.mlir --processor-assign="chip=bm1684x mode=INT8 num_device=1 num_core=1 addr_mode=auto high_precision=False" --import-calibration-table="file=yolov5s_cali_table asymmetric=False" --processor-top-optimize --convert-top-to-tpu=" asymmetric=False doWinograd=False q_group_size=0 q_symmetric=False matmul_perchannel=False gelu_mode=normal" --canonicalize --weight-fold -o yolov5s_bm1684x_int8_sym_tpu.mlir

../../build/bin/tpuc-opt yolov5s.mlir --processor-assign="chip=bm1684x mode=INT8 num_device=1 num_core=1 addr_mode=auto high_precision=False" -o yolov5s.mlir_processor-assign.mlir

../../build/bin/tpuc-opt yolov5s.mlir_processor-assign.mlir --import-calibration-table="file=yolov5s_cali_table asymmetric=False" -o yolov5s.mlir_calibration.mlir
../../build/bin/tpuc-opt yolov5s.mlir_calibration.mlir --processor-top-optimize -o yolov5s.mlir_processor-top-optimize.mlir
../../build/bin/tpuc-opt yolov5s.mlir_processor-top-optimize.mlir --convert-top-to-tpu=" asymmetric=False doWinograd=False q_group_size=0 q_symmetric=False matmul_perchannel=False gelu_mode=normal" -o yolov5s.mlir_convert-top-to-tpu.mlir

../../build/bin/tpuc-opt yolov5s.mlir_convert-top-to-tpu.mlir --canonicalize --weight-fold -o yolov5s.mlir_canonicalize.mlir
