# README

## 操作步骤


1. 下载模型。

```
sudoapt-getinstall git-lfs
git lfs install
code ~/.ssh/id_rsa.pub
https://huggingface.co/settings/keys

git clone git@hf.co:Qwen/Qwen2.5-VL-3B-Instruct-AWQ
Cloning into 'Qwen2.5-VL-3B-Instruct-AWQ'...
...
Filtering content: 100% (2/2), 3.17 GiB | 461.00 KiB/s, done.
```

2. 编译模型

```
./run.sh
```
