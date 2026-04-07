import os
import shutil
import tempfile
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEST_TEMP_ROOT = Path(__file__).resolve().parent.parent / "_pytest_tmp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

# Keep pytest temp dirs inside the repo because the global Windows temp root is
# intermittently inaccessible on this machine.
tempfile.tempdir = str(TEST_TEMP_ROOT)
os.environ["TMP"] = str(TEST_TEMP_ROOT)
os.environ["TEMP"] = str(TEST_TEMP_ROOT)
os.environ["TMPDIR"] = str(TEST_TEMP_ROOT)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def tmp_path():
    path = Path(tempfile.mkdtemp(prefix="pytest-", dir=TEST_TEMP_ROOT))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)