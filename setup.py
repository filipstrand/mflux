from setuptools import setup, find_packages

setup(
    name="mflux",
    version="0.2.0",
    author="Filip Strand",
    author_email="strand.filip@gmail.com",
    description="A MLX port of FLUX based on the Huggingface Diffusers implementation.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/filipstrand/mflux",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "mflux-generate=src.mflux.generate:main",
            "mflux-save=src.mflux.save:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.11',
    install_requires=[
        'mlx>=0.16.0',
        'numpy>=2.0.0',
        'pillow>=10.4.0',
        'transformers>=4.44.0',
        'sentencepiece>=0.2.0',
        'torch>=2.3.1',
        'tqdm>=4.66.5',
        'huggingface-hub>=0.24.5',
        'safetensors>=0.4.4',
        'piexif>=1.1.3',
    ],
)
