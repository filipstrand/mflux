This is a quick demo of `mflux` running on Linux GPUs in Replicate's cloud using its `cog` framework

# Development

- install a Docker runtime on your machine for the `cog build` step
- `cog login` if using for the first time
- `cog build --tag mflux-linux-cuda13:test`  # produces image cog-cloud-replicate-cog
- `cog predict  mflux-linux-cuda13:test --setup-timeout 3600 -i prompt=sunset -i steps=1 -i num_outputs=1 -i seed=$RANDOM`

when running the first time, we may need to allow a longer setup timeout for initial model download:

- `cog predict mflux-linux-cuda13:test --setup-timeout 3600 ...`
