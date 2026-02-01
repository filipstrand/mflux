This is the blocker with Modal B200 CUDA 13 mode:

```
==========
== CUDA ==
==========

CUDA Version 13.1.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Traceback (most recent call last):
  File "/pkg/modal/_runtime/container_io_manager.py", line 947, in handle_input_exception
    yield
  File "/pkg/modal/_container_entrypoint.py", line 172, in run_input_sync
    values = io_context.call_function_sync()
  File "/pkg/modal/_runtime/container_io_manager.py", line 225, in call_function_sync
    expected_value_or_values = self.finalized_function.callable(*args, **kwargs)
  File "/root/generate_remote.py", line 73, in generate_image
    from mflux.models.common.config import Config, ModelConfig
  File "/usr/local/lib/python3.13/site-packages/mflux/models/common/config/__init__.py", line 1, in <module>
    from mflux.models.common.config.config import Config
  File "/usr/local/lib/python3.13/site-packages/mflux/models/common/config/config.py", line 4, in <module>
    import mlx.core as mx
ImportError: cudaMemAdvise( data_, small_pool_size, cudaMemAdviseSetAccessedBy, loc) failed: invalid argument
Stopping app - uncaught exception raised in remote container: ImportError('cudaMemAdvise( data_, small_pool_size, cudaMemAdviseSetAccessedBy, loc) failed: invalid argument').
Exception ignored in atexit callback <nanobind.nb_func object at 0x2b2ded8517a0>:
RuntimeError: cudaMemAdvise( data_, small_pool_size, cudaMemAdviseSetAccessedBy, loc) failed: invalid argument
```
