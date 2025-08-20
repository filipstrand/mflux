This is the blocker with Linux CPU mode:

```
0%|          | 0/15 [10:12<?, ?it/s]
Traceback (most recent call last):
File "/pkg/modal/_runtime/container_io_manager.py", line 778, in handle_input_exception
  yield
File "/pkg/modal/_container_entrypoint.py", line 243, in run_input_sync
  res = io_context.call_finalized_function()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/pkg/modal/_runtime/container_io_manager.py", line 197, in call_finalized_function
  res = self.finalized_function.callable(*args, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/root/generate_remote.py", line 68, in generate_image
  image = flux.generate_image(
          ^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/site-packages/mflux/flux/flux.py", line 114, in generate_image
  mx.eval(latents)
RuntimeError: [AddMM::eval_cpu] Currently only supports float32.
```
