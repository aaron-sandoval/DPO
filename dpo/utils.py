from pathlib import Path

import torch as t


device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SEED: int = 1
LOW_GPU_MEM = device == t.device("cpu")