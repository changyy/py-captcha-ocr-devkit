"""CLI scaffolding tests for init and create-handler commands."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run_cli(cli_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(cli_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"CLI command failed with code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result


def test_init_copies_example_assets(temp_dir: Path, cli_path: Path) -> None:
    handlers_dir = temp_dir / "handlers"
    scripts_dir = temp_dir / "scripts"
    _run_cli(
        cli_path,
        "init",
        "--output-dir",
        str(handlers_dir),
        "--scripts-dir",
        str(scripts_dir),
        "--force",
    )

    expected_files = {
        "cnn_handler.py",
        "cnn_handler-requirements.txt",
        "cnn_handler-README.md",
        "cnn_handler-config.json",
        "crnn_handler.py",
        "crnn_handler-requirements.txt",
        "crnn_handler-README.md",
        "crnn_handler-config.json",
        "demo_handler.py",
        "demo_handler-README.md",
        "demo_handler-config.json",
        "ocr_common.py",
        "transformer_handler.py",
        "transformer_handler-requirements.txt",
        "transformer_handler-README.md",
        "transformer_handler-config.json",
        "__init__.py",
        "README.md",
    }

    actual_files = {item.name for item in handlers_dir.iterdir()}
    missing = expected_files - actual_files
    assert not missing, f"handlers init missing files: {sorted(missing)}"

    expected_scripts = {
        "api_cnn.sh",
        "api_crnn.sh",
        "api_transformer.sh",
        "evaluate_cnn.sh",
        "evaluate_crnn.sh",
        "evaluate_transformer.sh",
        "train_cnn.sh",
        "train_crnn.sh",
        "train_transformer.sh",
        "init.sh",
    }

    scripts_dir.mkdir(exist_ok=True)
    script_files = {item.name for item in scripts_dir.iterdir()}
    missing_scripts = expected_scripts - script_files
    assert not missing_scripts, f"init missing scripts: {sorted(missing_scripts)}"


def test_create_handler_generates_handler_and_readme(temp_dir: Path, cli_path: Path) -> None:
    output_dir = temp_dir / "custom_handlers"
    output_dir.mkdir()

    scripts_dir = temp_dir / "scripts"

    _run_cli(
        cli_path,
        "create-handler",
        "my_cnn",
        "--output-dir",
        str(output_dir),
        "--types",
        "preprocess,train,evaluate,ocr",
        "--scripts-dir",
        str(scripts_dir),
    )

    handler_file = output_dir / "my_cnn_handler.py"
    readme_file = output_dir / "my_cnn_handler-README.md"
    config_file = output_dir / "my_cnn_handler-config.json"

    assert handler_file.exists(), "Handler scaffold was not created"
    assert readme_file.exists(), "Handler README was not created"
    assert config_file.exists(), "Handler config was not created"

    config_text = config_file.read_text(encoding="utf-8")
    assert config_text.strip() == "{}", "Config file should default to empty JSON object"

    handler_source = handler_file.read_text(encoding="utf-8")
    assert "class MyCnnPreprocessHandler" in handler_source
    assert "class MyCnnOCRHandler" in handler_source
    assert 'HANDLER_ID = "my_cnn_preprocess"' in handler_source
    assert 'HANDLER_ID = "my_cnn_train"' in handler_source
    assert 'HANDLER_ID = "my_cnn_evaluate"' in handler_source
    assert 'HANDLER_ID = "my_cnn_ocr"' in handler_source

    readme_content = readme_file.read_text(encoding="utf-8")
    assert "# my_cnn Handler Blueprint" in readme_content
    assert "preprocess" in readme_content.lower()
    assert "ocr" in readme_content.lower()
    assert "my_cnn_handler-config.json" in readme_content

    expected_scripts = {
        scripts_dir / "train_my_cnn.sh",
        scripts_dir / "evaluate_my_cnn.sh",
        scripts_dir / "api_my_cnn.sh",
    }
    missing_scripts = [path for path in expected_scripts if not path.exists()]
    assert not missing_scripts, f"Missing generated scripts: {[str(p) for p in missing_scripts]}"

    train_script = (scripts_dir / "train_my_cnn.sh").read_text(encoding="utf-8")
    assert '"--handler" "my_cnn_train"' in train_script
    assert '"--handler-config" "my_cnn_train=${CONFIG}"' in train_script
    assert "CONFIG_ARGS=(" in train_script

    evaluate_script = (scripts_dir / "evaluate_my_cnn.sh").read_text(encoding="utf-8")
    assert '"--handler" "my_cnn_evaluate"' in evaluate_script

    api_script = (scripts_dir / "api_my_cnn.sh").read_text(encoding="utf-8")
    assert '"--handler" "my_cnn_ocr"' in api_script
    assert '"--preprocess-handler" "my_cnn_preprocess"' in api_script
