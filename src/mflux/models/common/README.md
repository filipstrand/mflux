### 🗜️ Quantization

MFLUX supports running FLUX in 3, 4, 5, 6, or 8-bit quantized mode. Running a quantized version can greatly speed up the
generation process and reduce the memory consumption by several gigabytes. [Quantized models also take up less disk space](#-size-comparisons-for-quantized-models). Simply pass the 

```sh
mflux-generate \
    --model schnell \
    --steps 2 \
    --seed 2 \
    --quantize 8 \
    --height 1920 \
    --width 1024 \
    --prompt "Tranquil pond in a bamboo forest at dawn, the sun is barely starting to peak over the horizon, panda practices Tai Chi near the edge of the pond, atmospheric perspective through the mist of morning dew, sunbeams, its movements are graceful and fluid — creating a sense of harmony and balance, the pond's calm waters reflecting the scene, inviting a sense of meditation and connection with nature, style of Howard Terpning and Jessica Rossier"
```

#### 💾 Saving a quantized version to disk

To save a local copy of the quantized weights, run the `mflux-save` command like so:

```sh
mflux-save \
    --path "/Users/filipstrand/Desktop/schnell_8bit" \
    --model schnell \
    --quantize 8
```

The `mflux-save` command works with both Flux and Qwen models.

*Note that when saving a quantized version, you will need the original huggingface weights.*

It is also possible to specify [LoRA](#-lora) adapters when saving the model, e.g

```sh
mflux-save \
    --path "/Users/filipstrand/Desktop/schnell_8bit" \
    --model schnell \
    --quantize 8 \
    --lora-paths "/path/to/lora.safetensors" \
    --lora-scales 0.7
```

When generating images with a model like this, no LoRA adapter is needed to be specified since
it is already baked into the saved quantized weights.

#### 💽 Loading and running a quantized version from disk

To generate a new image from the quantized model, simply provide a `--path` to where it was saved:

```sh
mflux-generate \
    --path "/Users/filipstrand/Desktop/schnell_8bit" \
    --model schnell \
    --steps 2 \
    --seed 2 \
    --height 1920 \
    --width 1024 \
    --prompt "Tranquil pond in a bamboo forest at dawn, the sun is barely starting to peak over the horizon, panda practices Tai Chi near the edge of the pond, atmospheric perspective through the mist of morning dew, sunbeams, its movements are graceful and fluid — creating a sense of harmony and balance, the pond's calm waters reflecting the scene, inviting a sense of meditation and connection with nature, style of Howard Terpning and Jessica Rossier"
```

*Note: When loading a quantized model from disk, there is no need to pass in `-q` flag, since we can infer this from the weight metadata.*

*Also Note: Once we have a local model (quantized [or not](#-running-a-model-directly-from-disk)) specified via the `--path` argument, the huggingface cache models are not required to launch the model.
In other words, you can reclaim the 34GB diskspace (per model) by deleting the full 16-bit model from the [Huggingface cache](#%EF%B8%8F-generating-an-image) if you choose.*

⚠️ * Quantized models saved with mflux < v.0.6.0 will not work with v.0.6.0 and later due to updated implementation. The solution is to [save a new quantized local copy](https://github.com/filipstrand/mflux/issues/149) 

*If you don't want to download the full models and quantize them yourself, the 4-bit weights are available here for a direct download:*
- For mflux < v.0.6.0:
  - [madroid/flux.1-schnell-mflux-4bit](https://huggingface.co/madroid/flux.1-schnell-mflux-4bit)
  - [madroid/flux.1-dev-mflux-4bit](https://huggingface.co/madroid/flux.1-dev-mflux-4bit)
- For mflux >= v.0.6.0:
  - [dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit](https://huggingface.co/dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit)
  - [dhairyashil/FLUX.1-dev-mflux-4bit](https://huggingface.co/dhairyashil/FLUX.1-dev-mflux-4bit)
  - [akx/FLUX.1-Kontext-dev-mflux-4bit](https://huggingface.co/akx/FLUX.1-Kontext-dev-mflux-4bit)
  - [filipstrand/FLUX.1-Krea-dev-mflux-4bit](https://huggingface.co/filipstrand/FLUX.1-Krea-dev-mflux-4bit)
  - [filipstrand/Qwen-Image-mflux-6bit](https://huggingface.co/filipstrand/Qwen-Image-mflux-6bit)
  - [filipstrand/Z-Image-Turbo-mflux-4bit](https://huggingface.co/filipstrand/Z-Image-Turbo-mflux-4bit)
  - [briaai/Fibo-mlx-4bit](https://huggingface.co/briaai/Fibo-mlx-4bit)
  - [briaai/Fibo-mlx-8bit](https://huggingface.co/briaai/Fibo-mlx-8bit)


Using the [community model support](#-third-party-huggingface-model-support), the quantized weights can be also be automatically downloaded when running the generate command:

```sh
mflux-generate \
    --model filipstrand/FLUX.1-Krea-dev-mflux-4bit \
    --base-model krea-dev \
    --prompt "A photo of a dog" \
    --steps 25 \
    --seed 2674888
```

```sh
mflux-generate-fibo \
    --model briaai/Fibo-mlx-4bit \
    --prompt-file ~/Desktop/bird.json \
    --width 1024 \
    --height 1024 \
    --steps 20 \
    --seed 42
```

⚠️ * Note: As of MFLUX v.0.13, some internal changes have been made which breaks compatibility with older pre-quantized models.
Newer ones will be uploaded, but in the meantime, you can always save a new quantized version from the original weights using the [mflux-save](#-saving-a-quantized-version-to-disk) command.*
To save disk space, you can delete the original full 16-bit model from the Huggingface cache after saving the quantized version.

---

### 💽 Running a model directly from disk

MFLUX supports running a model directly from a custom location using the `--model` flag with a local path.
In the example below, the model is placed in `/Users/filipstrand/Desktop/schnell`:

```sh
mflux-generate \
    --model "/Users/filipstrand/Desktop/schnell" \
    --base-model schnell \
    --steps 2 \
    --seed 2 \
    --prompt "Luxury food photograph"
```

When loading from a local path, use `--base-model` to specify the architecture (e.g., `schnell`, `dev`).

Also note that unlike when using the typical `alias` way of initializing the model (which internally handles that the required resources are downloaded),
when loading a model directly from disk, we require the downloaded models to look like the following:

<details>
<summary>📁 <strong>Required directory structure</strong></summary>

```
.
├── text_encoder
│   └── model.safetensors
├── text_encoder_2
│   ├── model-00001-of-00002.safetensors
│   └── model-00002-of-00002.safetensors
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── tokenizer_2
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── transformer
│   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   └── diffusion_pytorch_model-00003-of-00003.safetensors
└── vae
    └── diffusion_pytorch_model.safetensors
```

</details>

This mirrors how the resources are placed in the [HuggingFace Repo](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main) for FLUX.1.
*Huggingface weights, unlike quantized ones exported directly from this project, have to be
processed a bit differently, which is why we require this structure above.*

---

### 🌐 Third-Party HuggingFace Model Support

MFLUX supports compatible third-party models from HuggingFace that follow the FLUX architecture. The `--model` parameter accepts:

- **Predefined names**: `dev`, `schnell`, `fibo`, `z-image-turbo`, etc.
- **HuggingFace repos**: `Freepik/flux.1-lite-8B`, `briaai/Fibo-mlx-4bit`
- **Local paths**: `/Users/me/models/my-model`, `~/my-model`

This unified interface mirrors how LoRA paths work, making it easy to switch between local and remote models.

```sh
# Using a HuggingFace repo
mflux-generate \
    --model Freepik/flux.1-lite-8B \
    --base-model schnell \
    --steps 4 \
    --seed 42 \
    --prompt "A beautiful landscape with mountains and a lake"

# Using a local path
mflux-generate \
    --model /Users/me/models/flux-lite \
    --base-model schnell \
    --steps 4 \
    --seed 42 \
    --prompt "A beautiful landscape with mountains and a lake"
```

Some examples of compatible third-party models include:
- [Freepik/flux.1-lite-8B-alpha](https://huggingface.co/Freepik/flux.1-lite-8B-alpha) - A lighter version of FLUX
- [shuttleai/shuttle-3-diffusion](https://huggingface.co/shuttleai/shuttle-3-diffusion) - Shuttle's implementation based on FLUX

The model will be automatically downloaded from HuggingFace the first time you use it, similar to the official FLUX models.

*Note: Third-party models may have different performance characteristics, capabilities, or limitations compared to the official FLUX models. Always refer to the model's documentation on HuggingFace for specific usage instructions.*

---

### 🎨 Image-to-Image

One way to condition the image generation is by starting from an existing image and let MFLUX produce new variations.
Use the `--image-path` flag to specify the reference image, and the `--image-strength` to control how much the reference 
image should guide the generation. For example, given the reference image below, the following command produced the first
image using the  [Sketching](https://civitai.com/models/803456/sketching?modelVersionId=898364) LoRA: 

```sh
mflux-generate \
--prompt "sketching of an Eiffel architecture, masterpiece, best quality. The site is lit by lighting professionals, creating a subtle illumination effect. Ink on paper with very fine touches with colored markers, (shadings:1.1), loose lines, Schematic, Conceptual, Abstract, Gestural. Quick sketches to explore ideas and concepts." \
--image-path "reference.png" \
--image-strength 0.3 \
--lora-paths Architectural_Sketching.safetensors \
--lora-scales 1.0 \
--model dev \
--steps 20 \
--seed 43 \
--guidance 4.0 \
--quantize 8 \
--height 1024 \
--width 1024
```

Like with [Controlnet](#-controlnet), this technique combines well with [LoRA](#-lora) adapters:

![image](src/mflux/assets/img2img.jpg)

In the examples above the following LoRAs are used [Sketching](https://civitai.com/models/803456/sketching?modelVersionId=898364), [Animation Shot](https://civitai.com/models/883914/animation-shot-flux-xl-ponyrealism) and [flux-film-camera](https://civitai.com/models/874708?modelVersionId=979175) are used.

---

### 🔌 LoRA

MFLUX support loading trained [LoRA](https://huggingface.co/docs/diffusers/en/training/lora) adapters (actual training support is coming).

The following example [The_Hound](https://huggingface.co/TheLastBen/The_Hound) LoRA from [@TheLastBen](https://github.com/TheLastBen):

```sh
mflux-generate --prompt "sandor clegane" --model dev --steps 20 --seed 43 -q 8 --lora-paths "sandor_clegane_single_layer.safetensors"
```

![image](src/mflux/assets/lora1.jpg)
---

The following example is [Flux_1_Dev_LoRA_Paper-Cutout-Style](https://huggingface.co/Norod78/Flux_1_Dev_LoRA_Paper-Cutout-Style) LoRA from [@Norod78](https://huggingface.co/Norod78):

```sh
mflux-generate --prompt "pikachu, Paper Cutout Style" --model schnell --steps 4 --seed 43 -q 8 --lora-paths "Flux_1_Dev_LoRA_Paper-Cutout-Style.safetensors"
```
![image](src/mflux/assets/lora2.jpg)

*Note that LoRA trained weights are typically trained with a **trigger word or phrase**. For example, in the latter case, the sentence should include the phrase **"Paper Cutout Style"**.*

*Also note that the same LoRA weights can work well with both the `schnell` and `dev` models. Qwen models support LoRA but may require Qwen-specific LoRA weights. Refer to the original LoRA repository to see what mode it was trained for.*

#### Multi-LoRA

Multiple LoRAs can be sent in to combine the effects of the individual adapters. The following example combines both of the above LoRAs:

```sh
mflux-generate \
   --prompt "sandor clegane in a forest, Paper Cutout Style" \
   --model dev \
   --steps 20 \
   --seed 43 \
   --lora-paths sandor_clegane_single_layer.safetensors Flux_1_Dev_LoRA_Paper-Cutout-Style.safetensors \
   --lora-scales 1.0 1.0 \
   -q 8
```
![image](src/mflux/assets/lora3.jpg)

Just to see the difference, this image displays the four cases: One of having both adapters fully active, partially active and no LoRA at all.
The example above also show the usage of `--lora-scales` flag.

#### HuggingFace LoRA Downloads

MFLUX can automatically download LoRAs directly from HuggingFace. Simply pass the repository ID to `--lora-paths`:

```sh
# Download from a HuggingFace repo (auto-finds the .safetensors file)
mflux-generate \
    --prompt "a portrait" \
    --lora-paths "author/lora-model"
```

For repositories with multiple LoRA files (collections), use the `repo_id:filename` format to specify which file to download:

```sh
# Download a specific file from a collection
mflux-generate \
    --prompt "film storyboard style, a cat" \
    --lora-paths "ali-vilab/In-Context-LoRA:film-storyboard.safetensors"
```

Downloaded LoRAs are cached locally and reused on subsequent runs.

#### LoRA Library Path

MFLUX supports a convenient LoRA library feature that allows you to reference LoRA files by their basename instead of full paths. This is particularly useful when you have a collection of LoRA files organized in one or more directories.

To use this feature, set the `LORA_LIBRARY_PATH` environment variable to point to your LoRA directories. You can specify multiple directories separated by colons (`:`):

```sh
export LORA_LIBRARY_PATH="/path/to/loras:/another/path/to/more/loras"
```

Once set, MFLUX will automatically discover all `.safetensors` files in these directories (including subdirectories) and allow you to reference them by their basename:

```sh
# Instead of using full paths:
mflux-generate \
    --prompt "a portrait" \
    --lora-paths "/path/to/loras/style1.safetensors" "/another/path/to/more/loras/style2.safetensors"

# You can simply use basenames:
mflux-generate \
    --prompt "a portrait" \
    --lora-paths "style1" "style2"
```

<details>
<summary>Notes on organizing your LoRA files</summary>

- The basename is the filename without the `.safetensors` extension
- If multiple files have the same basename, the first directory in `LORA_LIBRARY_PATH` takes precedence
  - to workaround this, rename or symlink to another name your `.safetensors` files to avoid conflicts
- Full paths still work as before, making this feature fully backwards compatible
- The library paths are scanned recursively, so LoRAs in subdirectories are also discovered. However, we do not recommend setting the library paths to a directory with a large number of files, as it can slow down the scanning process on every run.
</details>

#### Supported LoRA formats (updated)

Since different fine-tuning services can use different implementations of FLUX, the corresponding
LoRA weights trained on these services can be different from one another. The aim of MFLUX is to support the most common ones.
The following table show the current supported formats:

| Supported | Name      | Example                                                                                                  | Notes                               |
|-----------|-----------|----------------------------------------------------------------------------------------------------------|-------------------------------------|
| ✅        | BFL       | [civitai - Impressionism](https://civitai.com/models/545264/impressionism-sdxl-pony-flux)                | Many things on civitai seem to work |
| ✅        | Diffusers | [Flux_1_Dev_LoRA_Paper-Cutout-Style](https://huggingface.co/Norod78/Flux_1_Dev_LoRA_Paper-Cutout-Style/) |                                     |
| ❌        | XLabs-AI  | [flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora/tree/main)                           |                                     |

To report additional formats, examples or other any suggestions related to LoRA format support, please see [issue #47](https://github.com/filipstrand/mflux/issues/47).
