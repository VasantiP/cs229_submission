import numpy as np
from scipy import stats

FEAT_CHANNEL_NAMES = ["RMSD", "Rg", "TM3-TM6"]
FEATURES_PER_FRAME = 3


def extract_summary_features(
    X_flat: np.ndarray,
    n_features_per_frame: int = FEATURES_PER_FRAME,
) -> "tuple[np.ndarray, list[str]]":
    n, total = X_flat.shape
    n_frames = total // n_features_per_frame
    bags = X_flat.reshape(n, n_frames, n_features_per_frame)

    feat_cols:  list[np.ndarray] = []
    feat_names: list[str]        = []

    for fi, cname in enumerate(FEAT_CHANNEL_NAMES):
        v = bags[:, :, fi]

        q10 = np.nanpercentile(v, 10, axis=1)
        q25 = np.nanpercentile(v, 25, axis=1)
        q50 = np.nanpercentile(v, 50, axis=1)
        q75 = np.nanpercentile(v, 75, axis=1)
        q90 = np.nanpercentile(v, 90, axis=1)

        moment_cols = [
            (np.nanmean(v, axis=1),                   f"{cname}_mean"),
            (np.nanstd(v,  axis=1),                   f"{cname}_std"),
            (np.nanmin(v,  axis=1),                   f"{cname}_min"),
            (np.nanmax(v,  axis=1),                   f"{cname}_max"),
            (np.nanmax(v,  axis=1) - np.nanmin(v, axis=1), f"{cname}_range"),
            (stats.skew(v, axis=1, nan_policy="omit").data
             if hasattr(stats.skew(v, axis=1, nan_policy="omit"), "data")
             else np.array(stats.skew(v, axis=1, nan_policy="omit")),
                                                       f"{cname}_skewness"),
            (np.array(stats.kurtosis(v, axis=1, nan_policy="omit")),
                                                       f"{cname}_kurtosis"),
            (np.nanmedian(v, axis=1),                  f"{cname}_median"),
            (q75 - q25,                                f"{cname}_IQR"),
            (q10,  f"{cname}_q10"),
            (q25,  f"{cname}_q25"),
            (q50,  f"{cname}_q50"),
            (q75,  f"{cname}_q75"),
            (q90,  f"{cname}_q90"),
        ]
        for col, name in moment_cols:
            feat_cols.append(np.asarray(col, dtype=np.float64))
            feat_names.append(name)

    cross_pairs = [
        (0, 1, "RMSD", "Rg"),
        (0, 2, "RMSD", "TM3-TM6"),
        (1, 2, "Rg",   "TM3-TM6"),
    ]
    for fi, fj, fn_i, fn_j in cross_pairs:
        vi = bags[:, :, fi]
        vj = bags[:, :, fj]
        corrs = np.array([
            np.corrcoef(vi[i], vj[i])[0, 1] if not np.any(np.isnan(vi[i]) | np.isnan(vj[i]))
            else np.nan
            for i in range(n)
        ])
        feat_cols.append(corrs)
        feat_names.append(f"corr_{fn_i}_{fn_j}")

    X_summary = np.column_stack(feat_cols)
    return X_summary, feat_names


def drop_nan_rows(
    X: np.ndarray, y: np.ndarray, label: str
) -> "tuple[np.ndarray, np.ndarray]":
    nan_mask = np.any(np.isnan(X), axis=1)
    if nan_mask.any():
        print(f"  Dropping {nan_mask.sum()} {label} samples with NaN "
              f"({nan_mask.sum()/len(y):.1%}).")
        X = X[~nan_mask]
        y = y[~nan_mask]
    return X, y
