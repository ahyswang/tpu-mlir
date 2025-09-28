

#tpuc-opt yolov5s_origin.mlir --shape-infer --canonicalize --extra-optimize -o yolov5s.mlir

tpuc-opt yolov5s_origin.mlir --shape-infer  -o yolov5s_origin_shape-infer.mlir
tpuc-opt yolov5s_origin_shape-infer.mlir --canonicalize -o yolov5s_origin_canonicalize.mlir
tpuc-opt yolov5s_origin_canonicalize.mlir --extra-optimize -o yolov5s_origin_extra-optimize.mlir

#tpuc-opt yolov5s.mlir --processor-assign="chip=bm1684x mode=F16 num_device=1 num_core=1 addr_mode=auto high_precision=False" --processor-top-optimize --convert-top-to-tpu=" asymmetric=False doWinograd=False q_group_size=0 q_symmetric=False matmul_perchannel=False gelu_mode=normal" --canonicalize --weight-fold -o yolov5s_bm1684x_f16_tpu.mlir

tpuc-opt yolov5s.mlir --processor-assign="chip=bm1684x mode=F16 num_device=1 num_core=1 addr_mode=auto high_precision=False" -o yolov5s_processor-assign.mlir
tpuc-opt yolov5s_processor-assign.mlir --processor-top-optimize  -o yolov5s_processor-processor-top-optimize.mlir
tpuc-opt yolov5s_processor-processor-top-optimize.mlir --convert-top-to-tpu=" asymmetric=False doWinograd=False q_group_size=0 q_symmetric=False matmul_perchannel=False gelu_mode=normal" -o yolov5s_convert-top-to-tpu.mlir

