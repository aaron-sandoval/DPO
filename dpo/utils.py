from pathlib import Path

import torch as t


device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SEED: int = 1
LOW_GPU_MEM = device == t.device("cpu")

def models_are_equal(model1: t.nn.Module, model2: t.nn.Module, epsilon: float = 1e-6) -> bool:
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    if len(params1) != len(params2):
        return False
    for p1, p2 in zip(params1, params2):
        if p1.shape != p2.shape:
            return False
        if not t.allclose(p1.data, p2.data, atol=epsilon):
            return False
    return True
