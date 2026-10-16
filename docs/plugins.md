# Plugins - User Manual

## What is a plugin?

A plugin is an optional package that installs next to mflux. It puts its code under `mflux.extras.<name>` and adds its own commands. The mflux commands do not change.

## Where to find plugins

The Plugins table in the [mflux-plugins README](https://github.com/mflux-community/mflux-plugins/blob/main/README.md) lists each plugin, its commands and its maintainer. Each plugin README tells which mflux versions it works with.

## How to install a plugin

Install mflux and the plugin together with one `uv` command. This example installs `mflux-teacache`:

```sh
uv tool install mflux --with-executables-from mflux-teacache
```

This gives you the mflux commands and the plugin commands side by side. If you install only the plugin with `uv tool install`, uv gives it its own environment with its own mflux. Then only the plugin commands go on your path.

In a Python environment that already has mflux, use pip:

```sh
pip install mflux-<name>
```

## How to see the plugin commands

`mflux-capabilities` lists the image commands of all packages in its environment, plugins included. To print only the command names, run:

```sh
mflux-capabilities --format markdown | grep "^## "
```

It lists a command only when the name starts with `mflux-generate`, `mflux-concept` or `mflux-upscale`. Shell completions show only the mflux commands.

## Where to report a problem

Run the plain mflux command with the same options. If it also fails, report the problem in the [mflux issues](https://github.com/mflux-community/mflux/issues). If only the plugin command fails, report it in the [mflux-plugins issues](https://github.com/mflux-community/mflux-plugins/issues).

## How to write a plugin

A plugin uses the command steps of an mflux command. These are the `validate`, `load` and `generate` methods and the `latent_creator` attribute of its command class, for example `ZImageCommand`. Only some mflux commands have them.

Follow these rules:

1. Put your code in one package under `mflux.extras`. Use a name that no other package uses.
2. Do not ship `mflux/__init__.py` or `mflux/extras/__init__.py`. The first file belongs to mflux. The second file hides all other plugins.
3. Call the `build_parser()` function and the command steps of the mflux command. Do not write your own. The guide tells you what to do when a part of `main()` has no step.
4. Test against mflux `main`. The command steps have no stability promise and can change in any release.

The [plugin guide](https://github.com/mflux-community/mflux-plugins/blob/main/docs/writing-a-plugin.md) gives the full procedure. Start from the [plugin template](https://github.com/mflux-community/mflux-plugins/tree/main/template).
