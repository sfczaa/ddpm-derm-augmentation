"""Dependency minimums for notebook checkpoint loading."""

from importlib.metadata import version

from packaging.version import Version


def require_training_runtime():
    minimums = {"torch": "2.13.0", "Pillow": "12.3.0"}
    for package, minimum in minimums.items():
        installed = version(package)
        if Version(installed) < Version(minimum):
            raise RuntimeError(
                f"{package}>={minimum} is required; installed {installed}. "
                "The notebook cannot continue in this runtime."
            )
