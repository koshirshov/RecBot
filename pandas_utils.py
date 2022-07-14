"""Utils for usage in model pipeline."""
import pandas as pd
import numpy as np
import pickle


def reduce_mem_usage(DF, verbose=True) -> pd.DataFrame:
    """Reduce mem by changing type of data.

    Args:
        DF (DataFrame): Dataframe to reduce memory.
        verbose (bool, optional): Verbose. Defaults to True.

    Returns:
        DataFrame: Reduced mem DF.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    e_16 = np.ceil(abs(np.log10(np.finfo(np.float16).eps)))
    e_32 = np.ceil(abs(np.log10(np.finfo(np.float32).eps)))
    start_mem = DF.memory_usage().sum() / 1024 ** 2
    for col in DF.columns:
        col_type = DF[col].dtypes
        stype = str(col_type)
        if col_type in numerics:
            c_min = DF[col].min()
            c_max = DF[col].max()
            if stype[:3] == "int":
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    DF[col] = DF[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    DF[col] = DF[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    DF[col] = DF[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    DF[col] = DF[col].astype(np.int64)
            else:
                l_max = len(sorted(DF[col].copy().astype(str), key=len)[-1])
                if l_max <= e_16:
                    DF[col] = DF[col].astype(np.float16)
                elif (e_16 < l_max <= e_32) and not stype.endswith("16"):
                    DF[col] = DF[col].astype(np.float32)
                elif not (stype.endswith("16") or stype.endswith("32")):
                    DF[col] = DF[col].astype(np.float64)
    end_mem = DF.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return DF


def save_data(PATH, data):
    """Save data locally."""
    with open(PATH, "wb") as f:
        pickle.dump(data, f)


def load_data(PATH) -> pd.DataFrame:
    """Load saved data."""
    with open(PATH, "rb") as f:
        data = pickle.load(f)
    return data
