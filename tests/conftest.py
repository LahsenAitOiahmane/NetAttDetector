import numpy as np
import pandas as pd
import pytest

from netattdetector.data import FEATURES


@pytest.fixture
def flows():
    rng = np.random.default_rng(123)
    rows = 600
    frame = pd.DataFrame(rng.lognormal(2, 1, (rows, len(FEATURES))), columns=FEATURES)
    target = np.arange(rows) % 3 == 0
    frame["IN_BYTES"] += target * 100
    frame["Label"] = target.astype(int)
    frame["Attack"] = np.where(target, "TestAttack", "Benign")
    frame["IPV4_SRC_ADDR"] = [f"host-{i % 15}" for i in range(rows)]
    frame["Timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="s").astype(str)
    return frame
