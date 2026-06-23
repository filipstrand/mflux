import pytest

from mflux.cli.completions.generator import CompletionGenerator


@pytest.mark.fast
def test_completion_generator_includes_fibo_edit_command():
    generator = CompletionGenerator()

    assert "mflux-generate-fibo-edit" in generator.commands

    parser = generator.create_parser_for_command("mflux-generate-fibo-edit")
    script = generator.generate_command_function("mflux-generate-fibo-edit", parser)

    assert "_mflux_generate_fibo_edit()" in script
    assert "--image-path" in script
    assert "--mask-path" in script
    assert "--prompt" in script


@pytest.mark.fast
def test_completion_generator_includes_krea2_command():
    generator = CompletionGenerator()

    assert "mflux-generate-krea2" in generator.commands

    parser = generator.create_parser_for_command("mflux-generate-krea2")
    script = generator.generate_command_function("mflux-generate-krea2", parser)

    assert "_mflux_generate_krea2()" in script
    assert "--model" in script
    assert "--lora-paths" in script
    assert "--lora-scales" in script
    assert "--image-path" in script
    assert "--prompt" in script
