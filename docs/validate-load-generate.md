# The validate, load and generate command steps

## Why plugins need three steps

A plugin cannot call `main()` of an mflux command. `main()` parses `sys.argv`, loads the model, generates the images and saves them in one function. A plugin must change one part of this flow and keep the other parts.

For this reason, a command that supports plugins puts its work in a command class, for example `QwenImageCommand`. The class has three static methods and one attribute:

| Member | Input | Output |
|---|---|---|
| `validate(args)` | parsed arguments | `ModelConfig` |
| `load(args)` | parsed arguments | the model object |
| `generate(model, args, seed, prompt)` | model, arguments, one seed, one prompt | `GeneratedImage` |
| `latent_creator` | not applicable | the latent creator class for callbacks |

`main()` of the command uses these same steps. Thus a plugin and the mflux command run the same code.

## validate

This step examines the request before the model loads. It does not read weights, so it is fast.

It does these tests:

1. It resolves `--model` against the model registry. It accepts only the models that this command can run. A wrong model name fails here.
2. Some commands also test the arguments. For example, some flags must occur together. The width and height must be multiples of 32. The output format must keep transparency.

It returns the `ModelConfig`. If a test fails, it raises an error.

## load

`load` makes the model object. It calls `validate` first, then gives the `ModelConfig` to the model class. It also gives the quantization, the model path, the LoRA arguments and, for some models, the compute precision.

This step reads the weights into memory. It is the slow step and uses the most memory. Call it one time and use the model again for each image.

There is one exception. If you register the callbacks with one seed and `--low-ram` or `--pid-decode`, the memory saver removes the transformer after the image. Call `load` again before the next `generate`.

Usually the model has three components:

- The text encoder. It changes the prompt into embeddings.
- The transformer. This is the diffusion model. It removes noise from the latents at each step.
- The VAE. It decodes the latents into the image.

Some models have more components. For example, FLUX.1 has two text encoders, an edit model has a vision encoder, and a ControlNet model has a ControlNet.

The load step does these tasks in this sequence:

1. It reads the weight files.
2. It loads the tokenizers.
3. It makes the components.
4. It puts the weights into the components. If you give `--quantize`, it quantizes them.
5. If you give `--lora`, it applies each LoRA to the transformer. By default, it merges the LoRA into the weights. Use `--no-bake-lora` to keep the LoRA separate.

## generate

`generate` makes one image from one seed and one prompt. It reads the other settings from `args`, for example the size, the steps, the guidance, the negative prompt and the input image paths. Then it calls `model.generate_image()` and returns a `GeneratedImage`.

`generate` applies the command defaults itself, for example the guidance value or the image size from a reference image. Thus a plugin that calls `generate` directly gets the same result as `main()`.

`generate` does not save the image. The caller saves it.

## How main() uses the steps

```python
model = QwenImageCommand.load(args)          # validate runs inside load
memory_saver = CallbackManager.register_callbacks(
    args=args, model=model, latent_creator=QwenImageCommand.latent_creator
)
for seed in args.seed:
    image = QwenImageCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
    image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
```

A plugin follows the same order. It can add work between the steps, or replace one step.

## Stability

The command steps have no stability promise. They can change in any mflux release. Test each plugin against mflux `main`. See [plugins.md](plugins.md) for the plugin rules.
