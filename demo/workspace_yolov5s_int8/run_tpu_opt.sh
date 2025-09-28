#!/bin/bash

# tpuc-opt yolov5s_bm1684x_f16_tpu.mlir --mlir-disable-threading --strip-io-quant="quant_input=False quant_output=False quant_input_list= quant_output_list= quant_output_bf16=False" --processor-tpu-optimize --dev-parallel --weight-reorder  --subnet-divide="dynamic=False" --op-reorder --future-update="rank=0 weight_list=" --layer-group="opt=2 group_by_cores=auto compress_mode=none debugger=0 disable_group_overlap=false"  --core-parallel  --address-assign -o yolov5s_bm1684x_f16_final.mlir --debug_cmd=

tpuc-opt yolov5s_bm1684x_f16_tpu.mlir --mlir-disable-threading --strip-io-quant="quant_input=False quant_output=False quant_input_list= quant_output_list= quant_output_bf16=False" --processor-tpu-optimize --dev-parallel --weight-reorder -o yolov5s_bm1684x_f16_tpu.mlir_weight_reorder.mlir

tpuc-opt yolov5s_bm1684x_f16_tpu.mlir_weight_reorder.mlir --subnet-divide="dynamic=False" -o yolov5s_bm1684x_f16_tpu.mlir_subnet.mlir

tpuc-opt yolov5s_bm1684x_f16_tpu.mlir_subnet.mlir --op-reorder --future-update="rank=0 weight_list=" -o yolov5s_bm1684x_f16_tpu.mlir_future_update.mlir

tpuc-opt yolov5s_bm1684x_f16_tpu.mlir_future_update.mlir --layer-group="opt=2 group_by_cores=auto compress_mode=none debugger=0 disable_group_overlap=false" -o yolov5s_bm1684x_f16_tpu.mlir_layer_group.mlir

tpuc-opt yolov5s_bm1684x_f16_tpu.mlir_layer_group.mlir --core-parallel -o yolov5s_bm1684x_f16_tpu.mlir_core_parallel.mlir

tpuc-opt yolov5s_bm1684x_f16_tpu.mlir_core_parallel.mlir --address-assign -o yolov5s_bm1684x_f16_tpu.mlir_address_assign.mlir

