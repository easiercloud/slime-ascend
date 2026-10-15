import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT = Path(__file__).with_name("prepare_test_resources.py")
SPEC = importlib.util.spec_from_file_location("prepare_test_resources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_convert_models_runs_from_selected_repo(tmp_path, monkeypatch):
    repo = tmp_path / "slime"
    hf_root = tmp_path / "hf"
    converted_root = tmp_path / "ma"
    repo.mkdir()
    (repo / "slime/utils/external_utils").mkdir(parents=True)
    (hf_root / "Model").mkdir(parents=True)
    converted_root.mkdir()

    observed = {}
    command_utils = ModuleType("slime.utils.external_utils.command_utils")

    def convert_checkpoint(**kwargs):
        observed["cwd"] = Path.cwd()
        (converted_root / "Model_torch_dist").mkdir()

    command_utils.convert_checkpoint = convert_checkpoint
    monkeypatch.setitem(sys.modules, "slime", ModuleType("slime"))
    monkeypatch.setitem(sys.modules, "slime.utils", ModuleType("slime.utils"))
    monkeypatch.setitem(sys.modules, "slime.utils.external_utils", ModuleType("slime.utils.external_utils"))
    monkeypatch.setitem(sys.modules, "slime.utils.external_utils.command_utils", command_utils)

    spec = MODULE.ModelSpec("Model", "model", 1, ("test.py",))
    result = MODULE.convert_models(repo, [spec], hf_root, converted_root)

    assert observed["cwd"] == repo
    assert result[0]["status"] == "converted"
