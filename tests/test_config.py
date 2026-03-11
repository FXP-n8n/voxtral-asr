import os
import pytest


def test_default_model_variant():
    os.environ.pop("MODEL_VARIANT", None)
    import importlib, config
    importlib.reload(config)
    assert config.MODEL_VARIANT == "mini"


def test_custom_model_variant():
    os.environ["MODEL_VARIANT"] = "small"
    import importlib, config
    importlib.reload(config)
    assert config.MODEL_VARIANT == "small"
    os.environ.pop("MODEL_VARIANT")


def test_default_upload_dir():
    os.environ.pop("UPLOAD_DIR", None)
    import importlib, config
    importlib.reload(config)
    assert config.UPLOAD_DIR == "./uploads"


def test_default_max_file_size():
    os.environ.pop("MAX_FILE_SIZE_MB", None)
    import importlib, config
    importlib.reload(config)
    assert config.MAX_FILE_SIZE_MB == 500


def test_default_host_port():
    import importlib, config
    importlib.reload(config)
    assert config.HOST == "0.0.0.0"
    assert config.PORT == 8000
