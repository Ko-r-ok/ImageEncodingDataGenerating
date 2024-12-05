from syntheval import SynthEval
import pandas as pd

# TODO save results to a file
# TODO to use syntheval benchmark as the final step
# TODO start writing the report

# loading the original data and the 3 fake data
original_data = pd.read_csv('diabetes.csv')
cgan_data = pd.read_csv('cgan.csv')
ctgan_data = pd.read_csv('ctgan.csv')
tvae_data = pd.read_csv('tvae.csv')

# loading the original into syntheval
evaluator = SynthEval(original_data)

# evaluate all 3 fake datasets
results_cgan = evaluator.evaluate(cgan_data, 'Outcome', presets_file = "full_eval")
results_ctgan = evaluator.evaluate(ctgan_data, 'Outcome', presets_file = "full_eval")
results_tvae = evaluator.evaluate(tvae_data, 'Outcome', presets_file = "full_eval")