# onnxparser-trt-plugin-sample

It's a sample for onnxparser working with trt user defined plugins for TRT7.0. 
It implements grid sample op in torch introduced in [this paper](https://arxiv.org/pdf/1506.02025.pdf)

# Purposes
This complemetary sample works like a collection of purposes. It would like to demonstrate:
1. How to export a custom op from torch via onnx
2. How to parse an onnx model which contains TensorRT custom op into TensorRT using onnxparser
3. How to enable dynamic shapes during this process
4. How to write plugin supporting explicit batch and dynamic shape for this process
5. How to write the python script to do so
   
# Dependency
This project depends on:
1. TensorRT 7.0
2. CUDA 10+
3. [TensorRT OSS](https://github.com/NVIDIA/TensorRT) project 

# Prepare, Build and Run
Follow instructions on TensorRT OSS project to prepare all env requirements. Make sure you can build TensorRT OSS project and run the sample.

**In order to build the project successfully, remember to checkout release 7.0 of TensorRT OSS and TensorRT/parser/onnx**.

You can choose to prepare this based on [NGC](https://ngc.nvidia.com) docker image tensorrt 20.01

Once you being able to run the built samples, you can patch files here into the TensorRT OSS project
1. copy TensorRT into TensorRT
2. copy symbolic_opset10.py to /path/to/python/dist-packages/torch/onnx/, say within ngc container, it's /user/local/lib/python3.6/dist-packages/torch/onnx/

After the pacthing, you should cd into TensorRT/parsers/onnx, follow instructions to build onnx parser, [link](https://github.com/onnx/onnx-tensorrt/blob/7.0/docker/onnx-tensorrt-tar.Dockerfile)

Once built onnx parser, remember to **make install** and **python setup.py build && python setup.py install** to get it taking effect. (If it's not working, check whether /usr/local/lib is in your path)

After that, rebuild TensorRT OSS project.

Done. Run LD_PRELOAD=/PATH/TO/YOUR/BUILT/libnvinfer_plugin.so python test_plugin_result.py to see if it's working

# How it works
Currently, to get torch-onnx-tensorrt working with custom op, you have to
1. Let torch/onnx recoganize and export the custom op that not in standard opset, here we choose opset10. Thus we need to hack into symbolic_opset10.py. Why we choose to
    modify symbolic_opset10.py? You can certainly call register_custom_op_symbolic() per https://github.com/onnx/tutorials/tree/master/PyTorchCustomOperator. But this function
    will require the original op is registered to a custom domain. While we have grid_sample in aten domain. In order to call register_custom_op_symbolic() to register your
    custom op, you have to write c code and recompile. This makes the solution even messier. It's not the best solution but workable. Note that all op registered in symbolic_opset10.py will be registered to onnx domain (this is forbidden in register_custom_op_symbolic() method).
2. Let onnxparser understand how to translate the custom op exists in onnx file to TensorRT layers including plugins. Thus we need to hack into builtin_op_importers.cpp
3. Let TensorRT know there's a new custom op named GridSampler. Here we implemented it as a plugin. Addtionally, since onnxparser only works with full dimension mode, we have to implement the trt plugin using IPluginV2DynamicExt introduced in TRT6.
4. Dynamic shape can be used if you export onnx model using dynamic_axes as illustrated in test_plugin_result.py. This will be processed in onnx-tensorrt module and then addInput is called using -1 as dummpy input dim. About how to build engine and context for dynamic shape, please refer to test_plugin_result.py.

# Limitations
1. This is not officially supported, thus an experimental sample.
2. Plugin namespace is not working.

# TODO
- [x] FP16 support
- [x] dynamic shape support in c++ sample
- [x] dynamic shape support in python sample working with onnxparser
- [x] FP16 support in python sample
- [ ] More tests to assure correctness
- [ ] 3D support in grid sample

