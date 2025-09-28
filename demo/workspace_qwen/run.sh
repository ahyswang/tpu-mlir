#!/bin/bash

llm_convert.py -m /workspace/Qwen2.5-VL-3B-Instruct-AWQ -s 2048 -q w4bf16 -c bm1684x --max_pixels 672,896 -o qwen2.5vl_3b
