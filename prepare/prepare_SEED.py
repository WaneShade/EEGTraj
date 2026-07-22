import argparse
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover - import guard for user envs
    raise ImportError(
        "prepare_SEED_from_mat.py requires scipy. Install scipy in the active environment first."
    ) from exc


DEFAULT_INPUT_ROOT = Path('/nvme/public/datasets/EEG/SEED/Preprocessed_EEG')
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'SEED'
DEFAULT_OUTPUT_NAME = 'seed-3.hdf5'
RSFREQ = 200
LFREQ = 0.0
HFREQ = 75.0
TRIAL_COUNT = 15
LABEL_MAP = {
    -1: 'S',
    0: 'N',
    1: 'H',
}
DEFAULT_CH_ORDER = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2',
]


class H5Dataset:
    def __init__(self, path: Path) -> None:
        self._file = h5py.File(path, 'w')

    def add_group(self, group_name: str):
        return self._file.create_group(group_name)

    def add_dataset(self, group: h5py.Group, dataset_name: str, arr: np.ndarray, chunks: tuple[int, int]):
        return group.create_dataset(dataset_name, data=arr, chunks=chunks)

    def add_attribute(self, src: 'h5py.Dataset|h5py.Group', attr_name: str, attr_value):
        src.attrs[attr_name] = attr_value

    def close(self):
        self._file.close()


def _string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding='utf-8'))


def _normalize_trial_array(arr: np.ndarray, mat_path: Path, key: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f'{mat_path.name}:{key} must be 2-D, got shape {arr.shape}')
    if arr.shape[0] == len(DEFAULT_CH_ORDER):
        out = arr
    elif arr.shape[1] == len(DEFAULT_CH_ORDER):
        out = arr.T
    else:
        raise ValueError(
            f'{mat_path.name}:{key} must contain 62 EEG channels, got shape {arr.shape}'
        )
    return np.asarray(out, dtype=np.float32)


def load_seed_labels(label_mat_path: Path) -> list[str]:
    label_mat = loadmat(label_mat_path)
    candidates = []
    for key, value in label_mat.items():
        if key.startswith('__'):
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size == TRIAL_COUNT and np.issubdtype(arr.dtype, np.number):
            candidates.append((key, arr.astype(int)))

    if not candidates:
        raise RuntimeError(f'Could not find a 15-element numeric label array in {label_mat_path}')

    preferred = next((arr for key, arr in candidates if key.lower() in {'label', 'labels'}), None)
    label_values = preferred if preferred is not None else candidates[0][1]
    try:
        return [LABEL_MAP[int(v)] for v in label_values]
    except KeyError as exc:
        raise RuntimeError(f'Unexpected SEED label value in {label_mat_path}: {exc}') from exc


def load_trials(mat_path: Path) -> list[np.ndarray]:
    mat = loadmat(mat_path)
    trials: dict[int, np.ndarray] = {}
    matched_keys: list[str] = []
    for key, value in mat.items():
        match = re.fullmatch(r'(?:[A-Za-z0-9]+_)?eeg_?(\d+)', key, flags=re.IGNORECASE)
        if match is None:
            continue
        trial_id = int(match.group(1))
        matched_keys.append(key)
        if trial_id in trials:
            raise RuntimeError(
                f'{mat_path.name} contains multiple arrays for trial {trial_id}: '
                f'{matched_keys}'
            )
        trials[trial_id] = _normalize_trial_array(value, mat_path=mat_path, key=key)

    missing = [idx for idx in range(1, TRIAL_COUNT + 1) if idx not in trials]
    if missing:
        visible_keys = sorted(key for key in mat.keys() if not key.startswith('__'))
        raise RuntimeError(
            f'{mat_path.name} is missing trial arrays: {missing}. '
            f'Matched keys: {sorted(matched_keys)}. Visible keys: {visible_keys}'
        )
    return [trials[idx] for idx in range(1, TRIAL_COUNT + 1)]


def pack_trials(trials: list[np.ndarray], sfreq: int) -> tuple[np.ndarray, list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    packed_trials = []
    cursor_sec = 0

    for idx, trial in enumerate(trials, start=1):
        n_samples = int(trial.shape[1])
        if n_samples % sfreq == 1:
            # Official SEED preprocessed trials sometimes include the segment end
            # point as an extra sample, yielding T * sfreq + 1 columns.
            print(
                f'  trial {idx}: trimming trailing endpoint sample '
                f'({n_samples} -> {n_samples - 1})'
            )
            trial = trial[:, :-1]
            n_samples = int(trial.shape[1])

        if n_samples % sfreq != 0:
            raise RuntimeError(
                f'Trial {idx} has {n_samples} samples, which is not divisible by sfreq={sfreq}'
            )
        trial_sec = int(n_samples // sfreq)
        starts.append(cursor_sec)
        ends.append(cursor_sec + trial_sec - 1)
        packed_trials.append(trial)
        cursor_sec += trial_sec

    eeg = np.concatenate(packed_trials, axis=1)
    return eeg, starts, ends


def discover_recordings(input_root: Path) -> list[tuple[int, str, Path]]:
    recordings = []
    for mat_path in sorted(input_root.glob('*.mat')):
        if mat_path.name.lower() == 'label.mat':
            continue
        match = re.fullmatch(r'(\d+)_(\d{8})', mat_path.stem)
        if match is None:
            continue
        subject_id = int(match.group(1))
        session_date = match.group(2)
        recordings.append((subject_id, session_date, mat_path))

    if not recordings:
        raise RuntimeError(
            f'No subject session .mat files matching "<subject>_<YYYYMMDD>.mat" were found in {input_root}'
        )
    recordings.sort(key=lambda item: (item[0], item[1]))
    return recordings


def build_hdf5(input_root: Path, output_path: Path) -> None:
    label_path = input_root / 'label.mat'
    if not label_path.exists():
        raise FileNotFoundError(f'Missing label file: {label_path}')

    labels = load_seed_labels(label_path)
    recordings = discover_recordings(input_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session_counters: dict[int, int] = defaultdict(int)
    chunks = (len(DEFAULT_CH_ORDER), RSFREQ)
    h5 = H5Dataset(output_path)
    try:
        for subject_id, session_date, mat_path in recordings:
            session_counters[subject_id] += 1
            session_idx = session_counters[subject_id]
            group_name = f'{subject_id:02d}_{session_idx}'

            print(f'processing {mat_path.name} -> {group_name}')
            trials = load_trials(mat_path)
            eeg, trial_starts, trial_ends = pack_trials(trials, sfreq=RSFREQ)

            group = h5.add_group(group_name)
            dataset = h5.add_dataset(group, 'eeg', eeg, chunks=chunks)

            h5.add_attribute(group, 'trialStart', np.asarray(trial_starts, dtype=np.int32))
            h5.add_attribute(group, 'trialEnd', np.asarray(trial_ends, dtype=np.int32))
            h5.add_attribute(group, 'label', _string_array(labels))
            h5.add_attribute(group, 'sourceFile', mat_path.name)
            h5.add_attribute(group, 'sessionDate', session_date)
            h5.add_attribute(group, 'subjectId', subject_id)
            h5.add_attribute(group, 'sessionIndex', session_idx)
            h5.add_attribute(group, 'timelineMode', 'packed_preprocessed_trials')

            h5.add_attribute(dataset, 'lFreq', LFREQ)
            h5.add_attribute(dataset, 'hFreq', HFREQ)
            h5.add_attribute(dataset, 'rsFreq', RSFREQ)
            h5.add_attribute(dataset, 'chOrder', _string_array(DEFAULT_CH_ORDER))
    finally:
        h5.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert official SEED Preprocessed_EEG .mat files to this repo\'s HDF5 format.'
    )
    parser.add_argument(
        '--input-root',
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f'Path to the official Preprocessed_EEG folder. Default: {DEFAULT_INPUT_ROOT}',
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
        help=f'Output HDF5 path. Default: {DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME}',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the output file if it already exists.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f'Input root does not exist: {input_root}')

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f'Output already exists: {output_path}. Re-run with --overwrite to replace it.'
        )

    build_hdf5(input_root=input_root, output_path=output_path)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    main()
