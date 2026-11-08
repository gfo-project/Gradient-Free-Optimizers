"""Integration tests for runnable example scripts in examples/."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent / "examples"

EXAMPLE_FILES = sorted(
    [p for p in EXAMPLES_DIR.rglob("*.py") if p.is_file()]
)

# Optional dependencies required by specific examples
OPTIONAL_DEPENDENCIES = {
    "_pytorch_optimizer.py": "torch",
    "deep_learning.py": "keras",
    "fit_gaussian.py": "matplotlib",
    "scipy_distributions.py": "scipy",
    "machine_learning.py": "sklearn",
    "memory.py": "sklearn",
}

ENV = os.environ.copy()
# Ensure matplotlib does not attempt to open a GUI window in test runs
ENV["MPLBACKEND"] = "Agg"


@pytest.mark.slow
@pytest.mark.parametrize(
    "example_path",
    EXAMPLE_FILES,
    ids=lambda p: str(p.relative_to(EXAMPLES_DIR).as_posix()),
)
def test_example_script(example_path: Path):
    """Verify that each example script executes successfully."""
    dep = OPTIONAL_DEPENDENCIES.get(example_path.name)
    if dep and importlib.util.find_spec(dep) is None:
        pytest.skip(
            f"Required dependency '{dep}' is not installed for {example_path.name}"
        )

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        env=ENV,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"Example {example_path.name} failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


if __name__ == "__main__":
    failed = []
    print(f"Running {len(EXAMPLE_FILES)} example scripts from {EXAMPLES_DIR}...\n")

    for path in EXAMPLE_FILES:
        name = path.name
        dep = OPTIONAL_DEPENDENCIES.get(name)
        if dep and importlib.util.find_spec(dep) is None:
            print(f"SKIP: {name} (requires '{dep}')")
            continue

        print(f"Testing {name}... ", end="", flush=True)
        try:
            res = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                env=ENV,
                timeout=180,
            )
            if res.returncode == 0:
                print("PASS")
            else:
                print(f"FAIL (exit code {res.returncode})")
                failed.append((name, res.stderr))
        except subprocess.TimeoutExpired:
            print("TIMEOUT (>180s)")
            failed.append((name, "Timeout expired"))

    if failed:
        print(f"\n{len(failed)} example(s) failed:")
        for name, err in failed:
            print(f"- {name}:\n{err}\n")
        sys.exit(1)
    else:
        print("\nAll tested examples passed!")
