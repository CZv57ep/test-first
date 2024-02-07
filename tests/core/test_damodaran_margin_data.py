import mergeron.core.damodaran_margin_data as dmd
import numpy as np
from scipy.stats import gaussian_kde

qtiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def test_data_generation() -> None:
    _obs, _wts, _stats = dmd.mgn_data_builder(dmd.scrape_data_table())

    _mgn_kde = gaussian_kde(_obs, weights=_wts)

    _ssz = 10**6
    _ssz_up = int(_ssz / (_mgn_kde.integrate_box_1d(0.0, 1.0) ** 2))
    print(_ssz_up)
    _sample_1 = _mgn_kde.resample(_ssz_up)[0]

    _sample_0 = _sample_1[:_ssz]
    print(np.quantile(_sample_0, qtiles))

    _sample_1 = _sample_1[(_sample_1 >= 0.0) & (_sample_1 <= 1)][:_ssz]
    assert len(_sample_1) == _ssz


if __name__ == "__main__":
    test_data_generation()
