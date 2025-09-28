#!/bin/bash

source $PWD/../../envsetup.sh

# run tpuc-opt tool.

# opt top mlir.
../../build/bin/tpuc-opt yolov5s_origin.mlir --shape-infer --canonicalize --extra-optimize -o yolov5s.mlir
# check top mlir result.
model_runner.py --input yolov5s_in_f32.npz --model ../yolov5s.onnx --output yolov5s_ref_outputs.npz
model_runner.py --input yolov5s_in_f32.npz --model yolov5s.mlir --output yolov5s_top_outputs.npz
npz_tool.py compare yolov5s_top_outputs.npz yolov5s_ref_outputs.npz --tolerance 0.99,0.99 --except - -vv

# convert top mlir to tpu mlir.
../../build/bin/tpuc-opt yolov5s.mlir --processor-assign="chip=bm1684x mode=INT8 num_device=1 num_core=1 addr_mode=auto high_precision=False" --import-calibration-table="file=yolov5s_cali_table asymmetric=False" --processor-top-optimize --convert-top-to-tpu=" asymmetric=False doWinograd=False q_group_size=0 q_symmetric=False matmul_perchannel=False gelu_mode=normal" --canonicalize --weight-fold -o yolov5s_bm1684x_int8_sym_tpu.mlir
# check tpu mlir result.
model_runner.py --input yolov5s_in_f32.npz --model yolov5s_bm1684x_int8_sym_tpu.mlir --output yolov5s_bm1684x_int8_sym_tpu_outputs.npz
npz_tool.py compare yolov5s_bm1684x_int8_sym_tpu_outputs.npz yolov5s_top_outputs.npz --tolerance 0.85,0.45 --except - -vv

# opt tpu mlir.
../../build/bin/tpuc-opt yolov5s_bm1684x_int8_sym_tpu.mlir --mlir-disable-threading --strip-io-quant="quant_input=False quant_output=False quant_input_list= quant_output_list= quant_output_bf16=False" --processor-tpu-optimize --dev-parallel --weight-reorder

# codegen bmodel.
../../build/bin/tpuc-opt yolov5s_bm1684x_int8_sym_final.mlir --codegen="model_file=yolov5s_1684x_int8.bmodel embed_debug_info=False model_version=latest bmodel_only=False gdma_check=True" -o /dev/null

# check bmodel result.
model_runner.py --input yolov5s_in_f32.npz --model yolov5s_1684x_int8.bmodel --output yolov5s_bm1684x_int8_sym_model_outputs.npz
npz_tool.py compare yolov5s_bm1684x_int8_sym_model_outputs.npz yolov5s_bm1684x_int8_sym_tpu_outputs.npz --tolerance 0.99,0.90 --except - -vv

