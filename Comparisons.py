# TODO hyperparameter optimization
# TODO compare with different architecture
# TODO add the mnist result to the report
# TODO maybe use some other image ganarator

import pandas as pd
from syntheval import SynthEval

original_data = pd.read_csv('datasets/diabetes.csv')
cgan_data = pd.read_csv('datasets/cgan_data.csv')
ctgan_data = pd.read_csv('datasets/ctgan_data.csv')
tvae_data = pd.read_csv('datasets/tvae_data.csv')



evaluator = SynthEval(original_data)

# evaluating all three datasets individually
cgan_results = evaluator.evaluate(cgan_data, 'Outcome', presets_file = "full_eval")
ctgan_results = evaluator.evaluate(ctgan_data, 'Outcome', presets_file = "full_eval")
tvae_results = evaluator.evaluate(tvae_data, 'Outcome', presets_file = "full_eval")

# running a benchmark across all datasets
df_vals, df_rank = evaluator.benchmark('datasets/','Outcome',rank_strategy='linear')

# saving every result into a file
cgan_results.to_csv('resutls/cgan_results.csv', index=False)
ctgan_results.to_csv('resutls/ctgan_results.csv', index=False)
tvae_results.to_csv('resutls/tvae_results.csv', index=False)
df_vals.to_csv('resutls/df_vals.csv', index=False)
df_rank.to_csv('resutls/df_rank.csv', index=False)