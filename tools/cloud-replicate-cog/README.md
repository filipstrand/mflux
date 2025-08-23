This is a quick demo of `mflux` running on Linux CPUs in Replicate's cloud using its `cog` framework

# Development

- `cog build --tag mflux-linux-cpu:test`  # produces image cog-cloud-replicate-cog
- `cog predict mflux-linux-cpu:test -i prompt=sunset -i steps=1 -i num_outputs=1 -i seed=$RANDOM`

when running the first time, we may need to allow a longer setup timeout for initial model download:

- `cog predict <image> --setup-timeout 3600 ...`
