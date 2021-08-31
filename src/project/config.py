"""
This is a configuration file that contains project wide constants used in 
code and in notebooks. Contains anything that might require some sort of
configuration before doing main analysis.
"""
from typing import List
import pandas as pd
import numpy as np


ENCODING = "ISO-8859-1"

IC_VARIABLES_ALL    = [
    "ic_cur_off_cde_1",
    "ic_cur_off_cde_2",
    "ic_cur_off_cde_3",
    "ic_prv_off_cde_1",
    "ic_prv_off_cde_2",
    "ic_prv_off_cde_3",
    "ic_escpe_hist_1",
    "ic_escpe_hist_2",
    "ic_escpe_hist_3",
    "ic_escpe_hist_4",
    "ic_escpe_hist_5",
    "ic_institut_adj",
    "ic_prior_commits",
    "ic_mths_to_release",
    "race",
    "sex",
    "ethnic_identity",
    "citizenship",
    "religion",
    "legal_zip_code",
    "ic_special_sent",
    "ic_mrtl_stat_fr_cl",
    "ic_employ_ind",
    "date_of_birth",
    "ic_prog_code_1",
    "ic_medical_cond",
    "ic_emotion_cond",
    "ic_da_ed",
    "ic_da_self_help",
    "ic_da_no_abuse",
    "ic_da_ongoing",
    "ic_da_therap",
    "ic_alcohol",
    "ic_drugs",
    "ic_da_score",
    "ic_ed_cond",
    "ic_voc_cond",
    "ic_othr_needs_cond",
    "ic_custdy_level",
    "ic_ovride_cust_lvl",
    "re_discip_reports",
    "control_number",
    "ic_date"
]

RE_VARIABLES_ALL    = [
    "re_curr_off_cd_1",
    "re_curr_off_cd_2",
    "re_curr_off_cd_3",
    "re_prev_off_cd_1",
    "re_prev_off_cd_2",
    "re_prev_off_cd_3",
    "re_escp_hist_1",
    "re_escp_hist_2",
    "re_escp_hist_3",
    "re_escp_hist_4",
    "re_escp_hist_5",
    "re_discip_reports",
    "re_age_for_class",
    "re_instit_violence",
    "ic_prior_commits",
    "race",
    "sex",
    "ethnic_identity",
    "citizenship",
    "religion",
    "legal_zip_code",
    "ic_employ_ind",
    "date_of_birth",
    "re_custody_level",
    "ic_custdy_level",
    "control_number",
    "re_de_year"
]




def build_subset_csv(df: pd.DataFrame, columns:List[str], save_filepath:str):
    '''
    Creates a new csv file that is a subset of the dataframe provided.

    New generated csv is created with only the specified columns. Strips
    all string columns so no trailing space characters remain in the new csv.

    Parameters
    ---
    df : pandas.DataFrame
        The main "superset" dataframe
    columns : List[str]
        The columns to keep
    save_filepath : str
        The location to save the generated csv
    '''
    csv_df = df[columns]
    
    for col in csv_df:
        try:
            csv_df.loc[:, col] = csv_df.loc[:, col].str.strip()
        except:
            pass
    
    csv_df.to_csv(save_filepath, index=False)


def concat_gs_prs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatanates the gravity and prior record scores to the specified dataframe.

    Translates the offense codes to the min and max grs, prs from the
    translation dataset. Does not modify the original dataframe.
    """
    df = df.copy()
    
    gs_prs_df = pd.read_csv(
        "../data/translate-gs-and-prs.csv",
        encoding=ENCODING
    )
    gs_prs_df.rename(
        columns={
            "perrec" : "off_cde",
            "max-gs" : "max_gs",
            "min-gs" : "min_gs",
            "max-prs" : "max_prs",
            "min-prs" : "min_prs"
        },
        inplace=True
    )

    def assign_gs(code, func=max, suffix="gs"):
        if isinstance(code, float):
            return np.NaN

        row = gs_prs_df[(gs_prs_df["off_cde"] == code)]
        if row.empty:
            # we didn't find an exact match
            # first let's check if a substring exists
            row = gs_prs_df[gs_prs_df["off_cde"].str.contains(code)]
        
        if row.empty:
            # didn't find a substring
            # let's take off a character and try to match again
            code = code[:-1]
        
            row = gs_prs_df[
                (gs_prs_df["off_cde"] == code) |
                gs_prs_df["off_cde"].str.contains(code)
            ]
            
            if row.empty:
                return np.NaN
        
        return func(
            max(row["max_{}".format(suffix)].values),
            min(row["min_{}".format(suffix)].values)
        )

    df["off_1_gs_max"] = df.loc[:, "ic_cur_off_cde_1"].apply(assign_gs, args=(max, "gs"))
    df["off_1_gs_min"] = df.loc[:, "ic_cur_off_cde_1"].apply(assign_gs, args=(min, "gs"))

    df["off_2_gs_max"] = df.loc[:, "ic_cur_off_cde_2"].apply(assign_gs, args=(max, "gs"))
    df["off_2_gs_min"] = df.loc[:, "ic_cur_off_cde_2"].apply(assign_gs, args=(min, "gs"))

    df["off_3_gs_max"] = df.loc[:, "ic_cur_off_cde_3"].apply(assign_gs, args=(max, "gs"))
    df["off_3_gs_min"] = df.loc[:, "ic_cur_off_cde_3"].apply(assign_gs, args=(min, "gs"))


    df["off_1_prs_max"] = df.loc[:, "ic_prv_off_cde_1"].apply(assign_gs, args=(max, "prs"))
    df["off_1_prs_min"] = df.loc[:, "ic_prv_off_cde_1"].apply(assign_gs, args=(min, "prs"))

    df["off_2_prs_max"] = df.loc[:, "ic_prv_off_cde_2"].apply(assign_gs, args=(max, "prs"))
    df["off_2_prs_min"] = df.loc[:, "ic_prv_off_cde_2"].apply(assign_gs, args=(min, "prs"))

    df["off_3_prs_max"] = df.loc[:, "ic_prv_off_cde_3"].apply(assign_gs, args=(max, "prs"))
    df["off_3_prs_min"] = df.loc[:, "ic_prv_off_cde_3"].apply(assign_gs, args=(min, "prs"))

    return df

def concat_gs_prs_re(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatanates the gravity and prior record scores to the specified dataframe.

    Translates the offense codes to the min and max grs, prs from the
    translation dataset. Does not modify the original dataframe.
    """
    df = df.copy()
    
    gs_prs_df = pd.read_csv(
        "../data/translate-gs-and-prs.csv",
        encoding=ENCODING
    )
    gs_prs_df.rename(
        columns={
            "perrec" : "off_cde",
            "max-gs" : "max_gs",
            "min-gs" : "min_gs",
            "max-prs" : "max_prs",
            "min-prs" : "min_prs"
        },
        inplace=True
    )

    def assign_gs(code, func=max, suffix="gs"):
        if isinstance(code, float):
            return np.NaN

        row = gs_prs_df[(gs_prs_df["off_cde"] == code)]
        if row.empty:
            # we didn't find an exact match
            # first let's check if a substring exists
            row = gs_prs_df[gs_prs_df["off_cde"].str.contains(code)]
        
        if row.empty:
            # didn't find a substring
            # let's take off a character and try to match again
            code = code[:-1]
        
            row = gs_prs_df[
                (gs_prs_df["off_cde"] == code) |
                gs_prs_df["off_cde"].str.contains(code)
            ]
            
            if row.empty:
                return np.NaN
        
        return func(
            max(row["max_{}".format(suffix)].values),
            min(row["min_{}".format(suffix)].values)
        )

    df["off_1_gs_max"] = df.loc[:, "re_curr_off_cd_1"].apply(assign_gs, args=(max, "gs"))
    df["off_1_gs_min"] = df.loc[:, "re_curr_off_cd_1"].apply(assign_gs, args=(min, "gs"))

    df["off_2_gs_max"] = df.loc[:, "re_curr_off_cd_2"].apply(assign_gs, args=(max, "gs"))
    df["off_2_gs_min"] = df.loc[:, "re_curr_off_cd_2"].apply(assign_gs, args=(min, "gs"))

    df["off_3_gs_max"] = df.loc[:, "re_curr_off_cd_3"].apply(assign_gs, args=(max, "gs"))
    df["off_3_gs_min"] = df.loc[:, "re_curr_off_cd_3"].apply(assign_gs, args=(min, "gs"))


    df["off_1_prs_max"] = df.loc[:, "re_prev_off_cd_1"].apply(assign_gs, args=(max, "prs"))
    df["off_1_prs_min"] = df.loc[:, "re_prev_off_cd_1"].apply(assign_gs, args=(min, "prs"))

    df["off_2_prs_max"] = df.loc[:, "re_prev_off_cd_2"].apply(assign_gs, args=(max, "prs"))
    df["off_2_prs_min"] = df.loc[:, "re_prev_off_cd_2"].apply(assign_gs, args=(min, "prs"))

    df["off_3_prs_max"] = df.loc[:, "re_prev_off_cd_3"].apply(assign_gs, args=(max, "prs"))
    df["off_3_prs_min"] = df.loc[:, "re_prev_off_cd_3"].apply(assign_gs, args=(min, "prs"))

    return df