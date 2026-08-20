import json
import os
from pathlib import Path

import numpy as np


def fake_pulsar_data(
    data_dir: Path, *psr_params: dict[str, float | np.ndarray]
) -> tuple[Path, ...]:
    """
    Create fake pulsar results, returning the directories in which they reside.

    Each entry in `psr_params` should be a dictionary that provides the following
    keys (value type given in brackets):
    - (float) TNRedGam
    - (float) TNRedAmp
    - (float) F2
    - (float) F0
    - (float) pepoch
    - (np.ndarray) mjd
    - (np.ndarray) residual

    The `mjd` and `residual` keys will be written to a `residuals.npz` file, whilst
    the other parameters will be written to a `model_params.json` file. Each
    dictionary provided will create these files and place them into a subdirectory
    within `tmp_path`. Subdirectories will be named as `psr-XX` with `XX` being the
    position in the `psr_params` sequence that the corresponding dictionary was given.

    Note that files are not automatically cleaned up by this function. This will need
    to be handled manually, or by wrapping in a temporary directory context.

    An error is raised if any of the above keys are missing, OR additional keys are
    present, for any of the parameter entries.

    :param data_dir: Directory into which to insert fake data.
    :param psr_params: Dictionaries containing data to write to pulsar directories.
    """
    expected_keys = {"TNRedGam", "TNRedAmp", "F2", "F0", "pepoch", "mjd", "residual"}
    subdirs_created = []

    for i, pulsar_data in enumerate(psr_params):
        present_keys = set(pulsar_data.keys())

        extra_keys = present_keys - expected_keys
        missing_keys = expected_keys - present_keys

        if extra_keys or missing_keys:
            msg = (
                "Extra keys: "
                + ", ".join(sorted(extra_keys))
                + "\nMissing keys: "
                + ", ".join(sorted(missing_keys))
            )
            raise KeyError(msg)

        # Create "pulsar" directory
        subdir = data_dir / f"psr-{i}"
        subdir.mkdir()
        subdirs_created.append(subdir)

        # Save .npz data
        np.savez(
            subdir / "residuals.npz",
            mjd=pulsar_data["mjd"],
            residual=pulsar_data["residual"],
        )

        # Save model parameters
        with Path.open(subdir / "model_params.json", "w") as f:
            # Dump everything except the array keys
            json.dump(
                {
                    key: value
                    for key, value in pulsar_data.items()
                    if key not in ("mjd", "residual")
                },
                f,
            )

    return tuple(subdirs_created)


def fake_concatenated_file(n_files: int, string: str = "PSR") -> Path:
    """
    Create a fake concatenated file, return path to the file.

    The concatenated file created here mimics several files starting with the same
    string having been concatenated.

    :param n_files: Number of files to concatenate.
    :param string: The string that indicates the begining of a file.
    """
    temp_path = os.environ.get("RUNNER_TEMP")
    path = Path(f"{temp_path}/data/fake/")
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"concat_{string}.txt"
    with path.open("w") as f:
        for i in range(n_files):
            f.write(string + "\n")
            f.write(f"some random text: {i}\n")
    return path
