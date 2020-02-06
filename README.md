# onnxparser-trt-plugin-sample
A sample for onnxparser working with trt user defined plugins for TRT7.0

# Dependency
This project depends on:
1. TensorRT 7.0
2. CUDA
3. [TensorRT OSS](https://github.com/NVIDIA/TensorRT) project 

# Prepare, Build and Run
Follow instructions on TensorRT OSS project to prepare all env requirements. Make sure you can build TensorRT OSS project and run the sample.

**In order to build the project successfully, remember to checkout release 7.0 of TensorRT OSS and TensorRT/parser/onnx**.

You can choose to prepare this based on [NGC](https://ngc.nvidia.com) docker image tensorrt 20.01

Once you being able to run the built samples, you can patch files here into the TensorRT OSS project
1. copy TensorRT into TensorRT
2. copy symbolic_opset10.py to /path/to/python/dist-packages/torch/onnx/, say within ngc container, it's /user/local/lib/python3.6/dist-packages/torch/onnx/

After the pacthing, you should cd into TensorRT/parsers/onnx, follow instructions to build onnx parser, [link](https://github.com/onnx/onnx-tensorrt/blob/7.0/docker/onnx-tensorrt-tar.Dockerfile)

Once built onnx parser, remember to *make install* and *python setup.py build && python setup.py install* to get it taking effect. (If it's not working, check whether /usr/local/lib is in your path)

After that, rebuild TensorRT OSS project.

Done. Run LD_PRELOAD=/PATH/TO/YOUR/BUILT/libnvinfer_plugin.so python test.py to see if it's working

# How it works
Currently, to get torch-onnx-tensorrt working with custom op, you have to
1. Let torch/onnx recoganize and export the custom op that not in standard opset, here we choose opset10. Thus we need to hack into symbolic_opset10.py
2. Let onnxparser understand how to translate the custom op exists in onnx file to TensorRT layers including plugins. Thus we need to hack into builtin_op_importers.cpp
3. Let TensorRT know there's a new custom op named GridSampler. Here we implemented it as a plugin. Addtionally, since onnxparser only works with full dimension mode, we have to implement the trt plugin using IPluginV2DynamicExt introduced in TRT6.

# Limitations
1. This is not officially supported, thus an experimental sample.
2. Dynamic shapes is not tested.
3. Plugin namespace is not working.

# Future
1. You can wait for future official release to support this feature.
2. GridSampler plugin's implementation will be released in following days.


