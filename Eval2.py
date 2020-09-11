from pathlib import Path

from Evaluator.PythonInterface import RegimeEvaluation
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools

index = 'SAP'
dynamic_exposure = False


def generate_analysis_file(regimes):
    num_regimes = regimes.shape[1]
    regimes_flat = pd.DataFrame(np.argmax(regimes.values, axis=1), index=regimes.index)
    regimes_flat.columns = ['policy']
    regime_mapping_perms = list(itertools.permutations(np.arange(num_regimes)))[1::]
    samples_collect = []
    for sample in regime_mapping_perms:
        regimes_flat_new = regimes_flat
        for i in sample:
            regimes_flat_new = regimes_flat_new.replace(i, sample[i])

        for value in range(num_regimes):
            regimes_flat_new[str(value)] = regimes_flat_new.apply(lambda row: int(row['policy']) == value, axis=1)
        regimes_flat_new = regimes_flat_new.drop('policy', axis=1)

        samples_collect += [regimes_flat_new]

    return samples_collect


def compute(file):
    _, policy_bin, i = str(file.stem).split('-')
    original_regime = pd.read_csv(file, index_col=0)
    generated_regimes = generate_analysis_file(original_regime)

    original_result, _ = RegimeEvaluation(original_regime, index, False, False).eval_period()
    original_result = original_result.T.iloc[[-1]]
    original_result = original_result.loc[:, ~original_result.columns.duplicated()]

    analysis_result = None
    for regimes in generated_regimes:
        evaluator = RegimeEvaluation(regimes, index, False, False)
        eval_result = evaluator.eval_period()
        result = eval_result[0].T.iloc[[-1]]
        result = result.loc[:, ~result.columns.duplicated()]

        analysis_result = result if analysis_result is None else pd.concat([analysis_result, result])

    return original_result, analysis_result, policy_bin, i


def main():
    folder = Path('Test')
    data_folder = folder / 'source'

    if not folder.exists():
        folder.mkdir(parents=True)

    pool = mp.Pool(mp.cpu_count())

    size = len(list(data_folder.iterdir()))
    for original_result, analysis_result, policy_bin, i in tqdm(pool.imap_unordered(compute, data_folder.iterdir()), total=size):
        original_result['model_id'] = f'Data-{policy_bin}-{i}'
        analysis_result['model_id'] = f'Data-{policy_bin}-{i}'

        eval_target = folder / f'Data_original-{policy_bin}.csv'
        analysis_target = folder / f'Data_analysis-{policy_bin}.csv'

        if eval_target.exists():
            original_result.to_csv(eval_target, mode='a', header=False, index=False)
            analysis_result.to_csv(analysis_target, mode='a', header=False, index=False)
        else:
            original_result.to_csv(eval_target, mode='w', header=True, index=False)
            analysis_result.to_csv(analysis_target, mode='w', header=True, index=False)

    # for file in data_folder.iterdir():
    #     original_result, analysis_result, policy_bin, i = compute(file)
    #
    #     original_result['model_id'] = f'Data-{policy_bin}-i'
    #     analysis_result['model_id'] = f'Data-{policy_bin}-i'
    #
    #     eval_target = folder / f'Data_original-{policy_bin}.csv'
    #     analysis_target = folder / f'Data_analysis-{policy_bin}.csv'
    #
    #     if eval_target.exists():
    #         original_result.to_csv(eval_target, mode='a', header=False, index=False)
    #         analysis_result.to_csv(analysis_target, mode='a', header=False, index=False)
    #     else:
    #         original_result.to_csv(eval_target, mode='w', header=True, index=False)
    #         analysis_result.to_csv(analysis_target, mode='w', header=True, index=False)


main()
