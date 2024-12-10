# TODO compare with different architecture
# TODO make sure it works with both datasets and the columns are correct
from syntheval import SynthEval

import Support

def evaluate(evaluator, location):
    dataframe = Support.read_file(location)
    return evaluator.evaluate(dataframe, 'Outcome', presets_file = "full_eval")

# first dataset
original_data = Support.read_file('datasets/diabetes.csv')
evaluator = SynthEval(original_data)

cgan_results_indians = evaluate(evaluator, 'datasets/cgan_data_indians.csv')
gan_results_indians = evaluate(evaluator, 'datasets/gan_data_indians.csv')
ctgan_results_indians = evaluate(evaluator, 'datasets/ctgan_data_indians.csv')
tvae_results_indians = evaluate(evaluator, 'datasets/tvae_data_indians.csv')

# second dataset
original_data = Support.read_file('datasets/diabetic_mellitus.arff')
evaluator = SynthEval(original_data)

cgan_results_mellitus = evaluate(evaluator, 'datasets/cgan_data_mellitus.csv')
gan_results_mellitus = evaluate(evaluator, 'datasets/gan_data_mellitus.csv')
ctgan_results_mellitus = evaluate(evaluator, 'datasets/ctgan_data_mellitus.csv')
tvae_results_mellitus = evaluate(evaluator, 'datasets/tvae_data_mellitus.csv')

# saving every result into a file
cgan_results_indians.to_csv('resutls/cgan_results_indians.csv', index=False)
gan_results_indians.to_csv('resutls/gan_results_indians.csv', index=False)
ctgan_results_indians.to_csv('resutls/ctgan_results_indians.csv', index=False)
tvae_results_indians.to_csv('resutls/tvae_results_indians.csv', index=False)
cgan_results_mellitus.to_csv('resutls/cgan_results_mellitus.csv', index=False)
gan_results_mellitus.to_csv('resutls/gan_results_mellitus.csv', index=False)
ctgan_results_mellitus.to_csv('resutls/ctgan_results_mellitus.csv', index=False)
tvae_results_mellitus.to_csv('resutls/tvae_results_mellitus.csv', index=False)

# running a benchmark across all datasets
# TODO make it work
df_vals, df_rank = evaluator.benchmark('datasets/','Outcome',rank_strategy='linear')

df_vals.to_csv('resutls/df_vals.csv', index=False)
df_rank.to_csv('resutls/df_rank.csv', index=False)