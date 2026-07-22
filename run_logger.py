import os
import sys
import time
import logging
from pathlib import Path

class TeeStream:
    """Write to both terminal and file-like stream."""
    def __init__(self, stream, file_stream):
        self.stream = stream
        self.file_stream = file_stream

    def write(self, data):
        self.stream.write(data)
        self.file_stream.write(data)

    def flush(self):
        self.stream.flush()
        self.file_stream.flush()

def setup_run_logging(log_dir: str, prefix: str = "train"):
    """Log to both terminal and a timestamped log file.

    Returns:
        log_path: path to the created log file.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}-{ts}.log")

    # 1) Python logging (optional usage via logging.info/debug/etc.)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 2) Mirror print() and tracebacks into the same file
    f = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)

    logging.info(f"[log] writing to: {log_path}")
    return log_path
