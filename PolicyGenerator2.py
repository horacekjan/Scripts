import sys
from typing import Union, List
import pandas as pd
import numpy as np
from pandas import DataFrame
from pathlib import Path


def generate_policies(monthly_turnover_rate_bins: List[List[int]], n_regimes: int, n_policies: int, index_size):
    n_year = 252
    n_years_in_data = int(index_size / n_year)
    random_policies = {}
    for monthly_turnover_rate_bin in monthly_turnover_rate_bins:
        assert monthly_turnover_rate_bin[1] <= 21, 'max bin rate can not be larger than the number of days in month'
        yearly_turnover_rate_bin = 12 * np.array(monthly_turnover_rate_bin)
        name = '_'.join([str(x) for x in monthly_turnover_rate_bin])
        random_policies[name] = []
        for _ in range(n_policies):
            yearly_turnover = np.random.uniform(*yearly_turnover_rate_bin)
            data_turnover = np.math.floor(n_years_in_data * yearly_turnover)
            regime_change_periods = np.random.choice(index_size, data_turnover, replace=False).tolist() + [0, index_size]
            regime_change_periods = sorted(list(set(regime_change_periods)))
            regimes = np.random.randint(n_regimes, size=len(regime_change_periods))
            policy = []
            for period_idx, (period_start, period_end) in enumerate(zip(regime_change_periods[:-1], regime_change_periods[1:])):
                for j in range(period_start, period_end):
                    policy.append(regimes[period_idx])
            random_policies[name].append(policy)
    return random_policies


def resolve_index(index):
    if type(index) == str:
        loaded_index = pd.read_csv(Path(__file__).parent / 'Data' / 'PolicyGenerator' / 'index.csv', index_col='DATE')
        if index == 'edc':
            return loaded_index['2011-07-01':'2013-06-28']
        elif index == 'gfc':
            return loaded_index['2008-07-01': '2010-06-30']
        elif index == 'eval_period':
            return loaded_index['2008-07-01':'2018-06-30']
        elif index == 'hold_out':
            return loaded_index['2018-07-01':'2020-03-02']
        else:
            print('Use custom index or one of: edc, gfc, eval_period, hold_out')
            sys.exit(1)
    elif isinstance(index, DataFrame):
        return index.index
    else:
        return index


def expand_policy(policy):
    unique_values = policy['policy'].unique()
    unique_values.sort()

    for value in unique_values:
        policy[str(value)] = policy.apply(lambda row: int(row['policy']) == value, axis=1)

    policy = policy.drop('policy', axis=1)

    return policy


def policy_report_generator(monthly_turnover_rate_bins: List[List[int]], data_frame_index: Union[str, DataFrame],
                            number_of_regimes: int, number_of_policies, index: str, dynamic_exposure: bool,
                            output_folder: Path):
    resolved_index = resolve_index(data_frame_index)

    policies = generate_policies(monthly_turnover_rate_bins, number_of_regimes, number_of_policies, len(resolved_index))

    for policy_bin, policies_list in policies.items():
        for i, policy in enumerate(policies_list):
            source_folder = output_folder / 'source'
            source_target = source_folder / f'Data-{policy_bin}-{i}.csv'
            if not source_folder.exists():
                source_folder.mkdir(parents=True)

            policy = pd.DataFrame(policy)
            policy.index = pd.to_datetime(resolved_index.index)
            policy.columns = ['policy']
            policy = expand_policy(policy)

            policy = policy.astype(int)

            policy.to_csv(source_target, mode='w', header=True, index=True)


if __name__ == '__main__':
    bins = [
        # SaP
        [12, 16],  # SAP main
        # [17, 21],
        # [22,26]

        # Topix
        # [16, 20],
        # [20, 21],  # TOPIX main
        # [25, 29],
        # [30, 34]
    ]

    n_of_regimes = 5  # 3
    n_of_policies = 100
    df_index = 'eval_period'  # hold_out
    target_index = 'sap'  # 'topix'
    dynamic_exp = True
    out_folder = Path('Test')

    policy_report_generator(bins, df_index, n_of_regimes, n_of_policies, target_index, dynamic_exp, out_folder)
