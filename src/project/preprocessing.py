from typing import Callable, Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_input_vars(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ic_date = df["ic_date"]//10000
    ic_date = ic_date.replace(0, np.nan)

    df["age"] = ic_date - (df.date_of_birth//10000)
    
    df["age_gt_45"] = ((ic_date - (df.date_of_birth//10000)\
        .astype(int)) > 45)\
        .apply(lambda x: 1 if x else 0)
    
    df["age_lt_25"] = ((ic_date - (df.date_of_birth//10000)\
        .astype(int)) < 25)\
        .apply(lambda x: 1 if x else 0)
    
    df["gender_female"] = df.sex.transform(lambda x: 1 if x == "F" else 0)

    df = pd.concat(
        [df, pd.get_dummies(df.race, prefix="race").drop("race_W", axis=1)],
        axis=1
    )
    
    df["prior_commits"] = df.ic_prior_commits

    df["escape_hist_1"] = df.ic_escpe_hist_1.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["escape_hist_2"] = df.ic_escpe_hist_2.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["escape_hist_3"] = df.ic_escpe_hist_3.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["escape_hist_4"] = df.ic_escpe_hist_4.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["escape_hist_5"] = df.ic_escpe_hist_5.transform(
        lambda x: 1 if str(x) == "X" else 0
    )

    df["escape_hist"] = (
        df["escape_hist_1"] +
        df["escape_hist_2"] +
        df["escape_hist_3"] +
        df["escape_hist_4"] +
        df["escape_hist_5"]
    )

    df = pd.concat(
        [
            df,
            pd.get_dummies(df.ic_mrtl_stat_fr_cl, prefix="mrt_stat")\
                .drop(["mrt_stat_SIN", "mrt_stat_UNK"], axis=1)
        ],
        axis=1
    )

    df["maritial_status"] = LabelEncoder().fit_transform(df["ic_mrtl_stat_fr_cl"])

    df["employed"] = df.ic_employ_ind.fillna(0).replace("X", 1)

    df["race"] = LabelEncoder().fit_transform(df["race"])
    
    return df

def preprocess_input_vars_re(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    re_date = df["re_de_year"]
    re_date = re_date.replace(0, np.nan)
    
    df["age"] = re_date - (df.date_of_birth//10000)

    df["age_gt_45"] = ((re_date - (df.date_of_birth//10000)\
        .astype(int)) > 45)\
        .apply(lambda x: 1 if x else 0)
    
    df["age_lt_25"] = ((re_date - (df.date_of_birth//10000)\
        .astype(int)) < 25)\
        .apply(lambda x: 1 if x else 0)
    
    df["gender_female"] = df.sex.transform(lambda x: 1 if x == "F" else 0)

    df = pd.concat(
        [df, pd.get_dummies(df.race, prefix="race").drop("race_W", axis=1)],
        axis=1
    )
    
    df["prior_commits"] = df.ic_prior_commits

    df["re_escp_hist_1"] = df.re_escp_hist_1.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["re_escp_hist_2"] = df.re_escp_hist_2.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["re_escp_hist_3"] = df.re_escp_hist_3.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["re_escp_hist_4"] = df.re_escp_hist_4.transform(
        lambda x: 1 if str(x) == "X" else 0
    )
    df["re_escp_hist_5"] = df.re_escp_hist_5.transform(
        lambda x: 1 if str(x) == "X" else 0
    )

    df["escape_hist"] = (
        df["re_escp_hist_1"] +
        df["re_escp_hist_2"] +
        df["re_escp_hist_3"] +
        df["re_escp_hist_4"] +
        df["re_escp_hist_5"]
    )

    df["employed"] = df.ic_employ_ind.fillna(0).replace("X", 1)
    
    return df

def col_values_by_custody(df:pd.DataFrame, col:str) -> Dict[int, List]:
    return dict(df.groupby("ic_custdy_level")[col].apply(list))


def perturb_data(df:pd.DataFrame,
                 target:str,
                 n:int,
                 sampling_funcs:Dict[str, Callable[[List[int], int], np.ndarray]]) -> pd.DataFrame:
    '''
    Perturbs the dataset and generates a new dataset from the original.

    Parameters
    ---
    df
        Original dataframe
    n
        The number of new samples per row of the original dataset
    target
        The target variable to group by
    sampling_funcs
        Dictionary of column names to functions that sample from the given enumeration
    '''
    col_enumerated = {
        col: dict(df.groupby(target)[col].apply(list))
        for col in df.drop(target, axis=1) 
    }

    default_func = sampling_funcs["default"]

    perturbed_data = df.apply(
        lambda row: np.array([
            sampling_funcs.get(col, default_func)(col_enumerated[col][row[target]], n)
            if col != target
            else [row[target] for _ in range(n)]
            for col in row.index
        ]).transpose(),
        axis=1
    )
    
    flattened = [row.tolist() for results in perturbed_data.values for row in results]

    return pd.DataFrame(flattened, columns=df.columns)
