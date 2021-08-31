from typing import Callable, Dict, List, Tuple
import itertools
from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

from . import rf_models


SamplingFnType = Callable[[pd.Series, int, str], pd.DataFrame]

# TODO: add some kind of display metrics or results function
class MCExp:
    def __init__(self, df:pd.DataFrame, prediction_var: str, rf:rf_models.BasicRFModel):
        self.df             = df
        self.prediction_var = prediction_var
        self.rf             = rf
        self.results        = None
    
    def run_sim(self, N:int, sampling_fn:SamplingFnType, ignore_vars:List[str]=[]):
        changes = {}
        drop_list = ignore_vars.copy()
        if not self.prediction_var in drop_list:
            drop_list.append(self.prediction_var)

        for variable in self.df[self.prediction_var].unique():
            data = self.df[self.df[self.prediction_var] == variable]

            # TODO: look at possibly using the `stack` method to achieve the
            # same thing
            perturbed_data = pd.concat([
                d
                for _, d in data.apply(
                    sampling_fn,
                    axis=1,
                    N=N,
                    pred_var=self.prediction_var
                ).iteritems()
            ], axis=0)

            prediction = self.rf.predict(perturbed_data.drop(drop_list, axis=1))
            change = prediction - variable
            control_num = perturbed_data.control_number
            
            changes[variable] = pd.DataFrame(data={
                "change": change,
                "control_number": control_num,
                "prediction": prediction
            })
        
        self.results = changes
    
    def print_results(self):
        if self.results is None:
            raise ValueError(
                "[print_results] No results available. Run simulation first."
            )
        for lvl, res in self.results.items():
            print(f"Level({lvl}):")
            for k, v in self.results[lvl]["change"].value_counts().iteritems():
                print(f"\t{k}: {v}")
    
    def plot_result_hist(self, normalize=False, title=None, save_to=None):
        if self.results is None:
            raise ValueError(
                "[plot_result_hist] No results available. Run simulation first."
            )
        
        fig, ax = plt.subplots(1, len(self.results), figsize=(16, 9), sharey=normalize)
        for i, lvl in enumerate(sorted(self.results.keys())):
            change_value_counts = self.results[lvl]["change"].value_counts(normalize=normalize)
            ax[i].bar(list(change_value_counts.index), list(change_value_counts.values))
            ax[i].set_xticks(list(change_value_counts.index))
            ax[i].set_xticklabels(list(change_value_counts.index))
            ax[i].set_title(f"Results for Custody Level {int(lvl)}")
        if title is not None:
            fig.suptitle(title)
        
        if save_to is not None:
            fig.savefig(save_to)
      
    def plot_mean_box(self, title=None, save_to=None):
        if self.results is None:
            raise ValueError(
                "[plot_result_box] No results available. Run simulation first."
            )

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        changes = {
            lvl: (data.groupby("control_number")["prediction"].mean() - lvl)
            for lvl, data in self.results.items()
        }
        
        ax.boxplot([ changes[k] for k in sorted(changes.keys()) ])
        ax.set_xlabel("Custody Level")
        ax.set_xticklabels(sorted(changes.keys()))
        if title is not None:
            fig.suptitle(title)
        
        if save_to is not None:
            fig.savefig(save_to)


class RepeatedReclassExp:
    def __init__(self, initial:pd.DataFrame, pred_var:str, rf:rf_models.BasicRFModel) -> None:
        self.rf = rf
        self.initial = initial
        self.prediction_variable = pred_var
        self.time_steps = None
    
    def run_reclassifications(self, N:int, ignore_vars:List[str]=[]):
        time = [ self.initial ] # this is time step 0
        droplist = ignore_vars.copy()
        if self.prediction_variable not in droplist:
            droplist.append(self.prediction_variable)

        for i in range(1, N):
            next_step = time[i - 1].copy(deep=True)
            next_step["ic_custdy_level"] = next_step["re_custody_level"]
            next_step["age"] = time[i - 1]["age"] + 1
            next_step["re_custody_level"] = self.rf.predict(time[i - 1].drop(droplist, axis=1))
            time.append(next_step)
        
        self.time_steps = time
    
    def plot_tragectories(self,
                          avg=True,
                          title=None,
                          xticklabels=False,
                          dif_linestyles=False,
                          save_to=None):
        if self.time_steps is None:
            raise ValueError(
                "[plot_tragectories] No timesteps found. Run reclassifications first."
            )
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        line_styles = ["-","--","-.",":"]
        markers = ["o", "v", "s", "*"]

        if not avg:
            # plot each person's trajectories
            for i in range(len(self.initial)):
                reclass_lvls = [
                    self.time_steps[t].iloc[i]["re_custody_level"]
                    for t in range(len(self.time_steps))
                ]
                if not dif_linestyles:
                    ax.plot(np.arange(0, len(reclass_lvls)), reclass_lvls)
                else:
                    # if only ploting in black/white
                    # use different linestyles for each custody level
                    # find which cust they originally belong to
                    orig_cust = int(self.initial.iloc[i]["re_custody_level"] - 2)
                    ax.plot(
                        np.arange(0, len(reclass_lvls)),
                        reclass_lvls,
                        line_styles[orig_cust],
                        # marker=markers[orig_cust],
                        c="black"
                    )
        else:
            # plot the mean of the trajectories of people who were initially at the same
            # reclass level

            # group people by their initial reclass level
            # and keep track of their control numbers
            ctrl_nums = {
                cust: self.initial[self.initial["re_custody_level"] == cust]["control_number"]
                for cust in self.initial["re_custody_level"].unique()
            }
            for cust in sorted(self.initial["re_custody_level"].unique()):
                reclass_lvls = []
                for t in range(len(self.time_steps)):
                    df = self.time_steps[t]
                    reclass_lvls.append(
                        df[df["control_number"].isin(ctrl_nums[cust])]["re_custody_level"].mean()
                    )
                if not dif_linestyles:
                    ax.plot(np.arange(0, len(reclass_lvls)), reclass_lvls)
                else:
                    ax.plot(
                        np.arange(0, len(reclass_lvls)),
                        reclass_lvls,
                        line_styles[int(cust) - 2],
                        # marker=markers[int(cust)],
                        c="black"
                    )

        if xticklabels:
            ax.set_xticks(np.arange(0, len(self.time_steps)))
            ax.set_xticklabels(np.arange(0, len(self.time_steps)))
        ax.set_yticks(np.arange(2, 6))
        ax.set_yticklabels(np.arange(2, 6))

        if title is not None:
            fig.suptitle(title)
        
        if save_to is not None:
            fig.savefig(save_to)
    
    def plot_timesteps_scatter(self, title=None):
        if self.time_steps is None:
            raise ValueError(
                "[plot_timesteps_scatter] No timesteps found. Run reclassifications first."
            )
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        for i in range(len(self.initial)):
            reclass_lvls = [
                self.time_steps[t].iloc[i]["re_custody_level"]
                for t in range(len(self.time_steps))
            ]
            ax.scatter(np.arange(0, len(reclass_lvls)), reclass_lvls)
        
        ax.set_xticks(np.arange(0, len(self.time_steps)))
        ax.set_xticklabels(np.arange(0, len(self.time_steps)))
        ax.set_yticks(np.arange(2, 6))
        ax.set_yticklabels(np.arange(2, 6))
        
        if title is not None:
            fig.suptitle(title)

    def count_total_changes(self):
        if self.time_steps is None:
            raise ValueError(
                "[count_total_changes] No timesteps found. Run reclassifications first."
            )
        
        return sum(sum([
            (self.time_steps[t]["re_custody_level"] - self.time_steps[t + 1]["re_custody_level"]).abs()
            for t in range(len(self.initial) - 1)
        ]))

    def calc_average_change_by_cust_per_person_per_year(self):
        if self.time_steps is None:
            raise ValueError(
                "[calculate_average_changes_per_person_per_year] No timesteps found. Run reclassifications first."
            )
        
        ctrl_nums = {
            cust: self.initial[self.initial["re_custody_level"] == cust]["control_number"]
            for cust in self.initial["re_custody_level"].unique()
        }

        changes_per_level = {}

        for cust, ctrl_num_list in ctrl_nums.items():
            changes_per_level[cust] = []
            for t in range(len(self.time_steps) - 1):
                current_lvl = self.time_steps[t].loc[
                    self.time_steps[t]["control_number"].isin(ctrl_num_list),
                    "re_custody_level"
                ]
                next_year_lvl = self.time_steps[t + 1].loc[
                    self.time_steps[t + 1]["control_number"].isin(ctrl_num_list),
                    "re_custody_level"
                ]

                changes_per_level[cust].append(np.sum((current_lvl - next_year_lvl).abs()))
        
        total_change_per_level = {
            lvl: np.sum(changes)
            for lvl, changes in changes_per_level.items()
        }

        return {
            lvl: (num_change / len(ctrl_nums[lvl])) / len(self.time_steps)
            for lvl, num_change in total_change_per_level.items()
        }


def gen_empirical_super_sampling_fn(col_enumerated: Dict,
                                    multi_lvl=False,
                                    consistency_cols:List[Tuple[str]]=[],
                                    const_cols:List[str]=[],
                                    group_cols:List[str]=None):
    '''
    Generates the corresponding empirical sampling function.

    col_enumerated - Dictionary of all values enumerated by custody level (or the variable that we
    are predicting) that appear for each column in dataset. The structure of the dictionary is:
        {
            "col_name": {
                1: [all values that appear for "col_name" at custody level 1],
                2: [- - - - - - - - || - -- - - - - - - - - - -  - - - - - 2],
                .
                .
                .
            }
            ...
        }
    
    multi_lvl - whether col_enumerated 2nd dictionary is multi-level. As in whether the enumeration
    is result of grouping multiple columns. If True, then structure of col_enumerated is expected to
    be:
        {
            "col_name": {
                (grouping): [all values that appear for "col_name" at custody level 1],
                (grouping): [- - - - - - - - || - -- - - - - - - - - - -  - - - - - 2],
                .
                .
                .
            }
            ...
        }
    
    '''
    
    def single_lvl_super_sampling(row: pd.Series, N: int, pred_var:str) -> pd.DataFrame:
        # randomly pick values from the enumerated list for all columns
        # except for the variable we are predicting for
        perturbed_data = {
            col: np.random.choice(col_enumerated[col][row[pred_var]], N)
            for col in row.index if col != pred_var and col not in const_cols
        }
        perturbed_data[pred_var] = np.array([row[pred_var] for _ in range(N)])
        for col in const_cols:
            perturbed_data[col] = np.array([row[col] for _ in range(N)])

        # do consistency checks
        for i in range(N):
            for consistency_grp in consistency_cols:
                num = sum([ perturbed_data[col][i] for col in consistency_grp ])
                if num > 1:
                    picklist = list(consistency_grp)
                    picklist.append(None)
                    pick = np.random.choice(picklist)

                    for col in consistency_grp:
                        perturbed_data[col][i] = 0
                    
                    if pick is not None:
                        perturbed_data[pick][i] = 1
        
        return pd.DataFrame(data=perturbed_data)
    
    def multi_lvl_super_sampling(row: pd.Series, N: int, pred_var:str):
        # randomly pick values from the enumerated list for all columns
        # except for the variable we are predicting for and the columns
        # that remain constant
        perturbed_data = {
            col: np.random.choice(
                col_enumerated[col][tuple([row[grp_col] for grp_col in group_cols])],
                N
            )
            for col in row.index
            if col != pred_var and col not in const_cols
        }
        perturbed_data[pred_var] = np.array([row[pred_var] for _ in range(N)])
        for col in const_cols:
            perturbed_data[col] = np.array([row[col] for _ in range(N)])
        
        # do consistency checks
        for i in range(N):
            for consistency_grp in consistency_cols:
                num = sum([ perturbed_data[col][i] for col in consistency_grp ])
                if num > 1:
                    picklist = list(consistency_grp)
                    picklist.append(None)
                    pick = np.random.choice(picklist)

                    for col in consistency_grp:
                        perturbed_data[col][i] = 0
                    
                    if pick is not None:
                        perturbed_data[pick][i] = 1
        
        return pd.DataFrame(data=perturbed_data)
    
    if not multi_lvl:
        return single_lvl_super_sampling
    
    return multi_lvl_super_sampling


def gen_empirical_one_col_sampling_fn(col_enumerated: Dict,
                                      multi_lvl=False,
                                      const_cols:List[str]=[],
                                      group_cols:List[str]=None):
    '''
    Generates the corresponding empirical sampling function.

    col_enumerated - Dictionary of all values enumerated by custody level (or the variable that we
    are predicting) that appear for each column in dataset. The structure of the dictionary is:
        {
            "col_name": {
                1: [all values that appear for "col_name" at custody level 1],
                2: [- - - - - - - - || - -- - - - - - - - - - -  - - - - - 2],
                .
                .
                .
            }
            ...
        }
    
    multi_lvl - whether col_enumerated 2nd dictionary is multi-level. As in whether the enumeration
    is result of grouping multiple columns. If True, then structure of col_enumerated is expected to
    be:
        {
            "col_name": {
                (grouping): [all values that appear for "col_name" at custody level 1],
                (grouping): [- - - - - - - - || - -- - - - - - - - - - -  - - - - - 2],
                .
                .
                .
            }
            ...
        }
    
    '''
    
    def single_lvl_one_col_sampling(row: pd.Series, N: int, pred_var:str) -> pd.DataFrame:
        # randomly pick values from the enumerated list for all columns
        # except for the variable we are predicting for
        data = {
            col: np.array([row[col] for _ in range(N)])
            for col in row.index
        }
        perturbed_data = {
            col: np.random.choice(col_enumerated[col][row[pred_var]], N)
            for col in row.index if col != pred_var and col not in const_cols
        }
        
        # pick random column to change
        # and then perturb accordingly
        rand_cols = np.random.choice(list(perturbed_data.keys()), size=N)
        
        for i, col_to_perturb in enumerate(rand_cols):
            data[col_to_perturb][i] = perturbed_data[col_to_perturb][i]
            
        return pd.DataFrame(data=data)
    
    def multi_lvl_one_col_sampling(row: pd.Series, N: int, pred_var:str):
        # randomly pick values from the enumerated list for all columns
        # except for the variable we are predicting for
        data = {
            col: np.array([row[col] for _ in range(N)])
            for col in row.index
        }
        perturbed_data = {
            col: np.random.choice(
                col_enumerated[col][tuple([row[grp_col] for grp_col in group_cols])],
                N
            )
            for col in row.index if col != pred_var and col not in const_cols
        }
        
        # pick random column to change
        # and then perturb accordingly
        rand_cols = np.random.choice(list(perturbed_data.keys()), size=N)
        
        for i, col_to_perturb in enumerate(rand_cols):
            data[col_to_perturb][i] = perturbed_data[col_to_perturb][i]
            
        return pd.DataFrame(data=data)

    if not multi_lvl:
        return single_lvl_one_col_sampling
    
    return multi_lvl_one_col_sampling


def gen_95conf_super_sampling_fn(col_enumerated: Dict,
                                 conf_level=0.95,
                                 multi_lvl=False,
                                 const_cols:List[str]=[],
                                 group_cols:List[str]=None):
    '''
    Works and has same assumptions as other gen_ functions
    '''
    def single_lvl_95_super_sampling(row: pd.Series, N:int, pred_var:str):
        perturbed_data = {}
        for col in row.index:
            if col == pred_var or col in const_cols:
                continue

            conf_interval = stats.t.interval(
                conf_level,
                len(col_enumerated[col][row[pred_var]]) - 1,
                row[col],
                stats.sem(col_enumerated[col][row[pred_var]])
            )
            perturbed_data[col] = np.random.uniform(conf_interval[0], conf_interval[1], N)
            # perturbed_data[col] = np.random.normal(
            #     row[col],
            #     stats.sem(col_enumerated[col][row[pred_var]]),
            #     N
            # )

        
        perturbed_data[pred_var] = np.array([row[pred_var] for _ in range(N) ])
        for col in const_cols:
            perturbed_data[col] = np.array([row[col] for _ in range(N) ])
        
        return pd.DataFrame(data=perturbed_data)
    
    def multi_lvl_95_super_sampling(row: pd.Series, N:int, pred_var:str):
        perturbed_data = {}
        for col in row.index:
            if col == pred_var or col in const_cols:
                continue

            conf_interval = stats.t.interval(
                conf_level,
                len(col_enumerated[col][tuple([row[grp_col] for grp_col in group_cols])]) - 1,
                np.nanmean(col_enumerated[col][tuple([row[grp_col] for grp_col in group_cols])]),
                stats.sem(col_enumerated[col][tuple([row[grp_col] for grp_col in group_cols])])
            )
            if np.isnan(conf_interval):
                print(f"We got NAN: {conf_interval}")

            data_points = np.arange(conf_interval[0], conf_interval[1])
            perturbed_data[col] = np.random.choice(data_points, N)
        
        perturbed_data[pred_var] = np.array([row[pred_var] for _ in range(N) ])
        for col in const_cols:
            perturbed_data[col] = np.array([row[col] for _ in range(N) ])
        
        return pd.DataFrame(data=perturbed_data)
    
    if multi_lvl:
        return multi_lvl_95_super_sampling
    
    return single_lvl_95_super_sampling

def sensitivity_analysis(df:pd.DataFrame,
                         perturb_vars:List[str],
                         rf:rf_models.BasicRFModel,
                         drop_vars:List[str]=[],
                         prct_pertub:float=0.1):
    assert prct_pertub >= 0 and prct_pertub <= 1

    results_df = df.copy(deep=True)
    results_df["original_pred"] = rf.predict(df.drop(drop_vars, axis=1))

    for col in perturb_vars:
        new_df = df.copy(deep=True)
        # original_col = new_df[col]
        # perturb up and see result
        new_df[col] = df[col] * (1 + prct_pertub)
        results_df[f"{col}_up_pred"] = rf.predict(new_df.drop(drop_vars, axis=1))
        # new_df[col] = original_col
        # perturb down and see result
        new_df[col] = df[col] * (1 - prct_pertub)
        results_df[f"{col}_down_pred"] = rf.predict(new_df.drop(drop_vars, axis=1))
        # new_df[col] = original_col
    
    return results_df

