import platform
import subprocess


class AppleSiliconUtil:
    _chip_name: str | None = None

    @classmethod
    def is_m1_or_m2(cls) -> bool:
        if platform.system() != "Darwin":
            return False
        if platform.machine() not in {"arm64", "aarch64"}:
            return False
        chip_name = cls._get_chip_name().lower()
        if "max" in chip_name or "ultra" in chip_name:
            return False
        return "apple m1" in chip_name or "apple m2" in chip_name

    @classmethod
    def should_use_compile(cls) -> bool:
        """Return True only when mx.compile is known to be safe on this chip.

        M1 and M2 (including Max/Ultra variants) trigger a C++ unordered_map::at
        key-not-found crash in MLX when mx.compile is used with large LoRA graphs
        and mx.clear_cache() is called between denoising steps.
        M3 and later do not exhibit this issue.
        """
        if platform.system() != "Darwin":
            return True
        chip_name = cls._get_chip_name().lower()
        # "apple m1" matches M1 / M1 Max / M1 Ultra
        # "apple m2" matches M2 / M2 Max / M2 Ultra
        return not ("apple m1" in chip_name or "apple m2" in chip_name)

    @classmethod
    def _get_chip_name(cls) -> str:
        if cls._chip_name is not None:
            return cls._chip_name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            cls._chip_name = result.stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            cls._chip_name = ""
        return cls._chip_name
