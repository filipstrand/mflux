def pytest_configure(config):
    config.addinivalue_line("markers", "fast: marks tests as fast (no image generation)")
    config.addinivalue_line("markers", "slow: marks tests as slow (generates images)")
