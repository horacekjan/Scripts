from pathlib import Path

INPUT_DIRECTORY = Path(r'D:\Work\Tmp\Roman\Sensitivity\report')
OUTPUT_DIRECTORY = Path(r'D:\Work\Tmp\Roman\Sensitivity\result\0')

if not OUTPUT_DIRECTORY.exists():
    OUTPUT_DIRECTORY.mkdir(parents=True)

for folder in INPUT_DIRECTORY.iterdir():
    input_file = folder / 'test' / 'regimes'
