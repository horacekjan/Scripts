from pathlib import Path

from Evaluator.PythonInterface import RegimeEvaluation
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

index = 'SAP'
dynamic_exposure = True


def compute(file):
    _, policy_bin, i = str(file.stem).split('-')
    evaluator = RegimeEvaluation(file, index, False, dynamic_exposure)
    eval_result = evaluator.edc()

    result = eval_result[0].T.iloc[[-1]]
    for name in sorted(eval_result[1][file.stem].keys()):
        value = eval_result[1][file.stem][name]
        post_fix = name.split('_')[-1]
        result_line = value.iloc[[-1]]
        columns = [f'{x}_{post_fix}' for x in result_line.columns]
        result_line.columns = columns
        result = pd.concat([result, result_line], axis=1, sort=False)

    result = result.loc[:, ~result.columns.duplicated()]

    return result, policy_bin, i


def main():
    data_folder = Path('aws_data')
    output_folder = Path('aws_output')
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    pool = mp.Pool(mp.cpu_count())

    size = len(list(data_folder.iterdir()))
    for result, policy_bin, i in tqdm(pool.imap_unordered(compute, data_folder.iterdir()), total=size):
        eval_target = output_folder / Path(f'Data-{policy_bin}.csv')

        if eval_target.exists():
            result.to_csv(eval_target, mode='a', header=False, index=False)
        else:
            result.to_csv(eval_target, mode='w', header=True, index=False)


main()