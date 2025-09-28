#!/bin/bash

python main.py

# model input shape = (4,3,32,32)

model_transform.py \
--model_name dynamic_shape_model \
--model_def dynamic_shape_model.onnx \
--input_shapes [[4,3,32,32]] \
--mean 0.0,0.0,0.0  \
--scale 0.0039216,0.0039216,0.0039216 \
--keep_aspect_ratio \
--pixel_format rgb \
--output_names output \
--mlir dynamic_shape_model.mlir --dynamic > python.log 2>&1

model_deploy.py \
  --mlir dynamic_shape_model.mlir \
  --quantize F16 \
  --processor bm1684x \
  --dynamic \
  --model dynamic_shape_model_1684x_f16.bmodel >> python.log 2>&1

# forward input shape = (2,3,32,32)

model_runner.py --input dynamic_shape_input.npz --model dynamic_shape_model.onnx --output dynamic_shape_output_onnx.npz
npz_tool.py compare dynamic_shape_output_onnx.npz dynamic_shape_output.npz --tolerance 0.99,0.90 --except - -vv

model_runner.py --input dynamic_shape_input.npz --model dynamic_shape_model.mlir --output dynamic_shape_output_top.npz
npz_tool.py compare dynamic_shape_output_top.npz dynamic_shape_output.npz --tolerance 0.99,0.90 --except - -vv

# crash
# model_runner.py --input dynamic_shape_input.npz --model dynamic_shape_model_bm1684x_f16_tpu.mlir --output dynamic_shape_output_tpu.npz
# npz_tool.py compare dynamic_shape_output_tpu.npz dynamic_shape_output.npz --tolerance 0.99,0.90 --except - -vv

# forward input shape = (2,3,32,32)

model_runner.py --input dynamic_shape_input.npz --model dynamic_shape_model_1684x_f16.bmodel --output dynamic_shape_output_1684x_f16.npz
npz_tool.py compare dynamic_shape_output_1684x_f16.npz dynamic_shape_output.npz --tolerance 0.99,0.90 --except - -vv
