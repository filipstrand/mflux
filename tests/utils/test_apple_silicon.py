from mflux.utils.apple_silicon import AppleSiliconUtil


def test_is_m1_or_m2_true_for_m1_and_m2_non_max_variants(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr(AppleSiliconUtil, "_get_chip_name", lambda: "Apple M2 Pro")

    assert AppleSiliconUtil.is_m1_or_m2() is True


def test_is_m1_or_m2_false_for_m2_max(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr(AppleSiliconUtil, "_get_chip_name", lambda: "Apple M2 Max")

    assert AppleSiliconUtil.is_m1_or_m2() is False


def test_is_m1_or_m2_false_for_m2_ultra(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr(AppleSiliconUtil, "_get_chip_name", lambda: "Apple M2 Ultra")

    assert AppleSiliconUtil.is_m1_or_m2() is False


def test_is_m1_or_m2_false_outside_darwin(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr(AppleSiliconUtil, "_get_chip_name", lambda: "Apple M2")

    assert AppleSiliconUtil.is_m1_or_m2() is False
