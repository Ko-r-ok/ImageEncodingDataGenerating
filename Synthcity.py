from synthcity.plugins import Plugins
import pandas as pd

import Support

# generate data with synthcity and transform it to a normal dataframe -- this is a mess
def handle_synthcitys_mess(model, amount):
    gen_data = model.generate(count=amount).unpack()
    gen_labels = gen_data[1]
    gen_data = gen_data[0]
    gen_labels = pd.DataFrame({'Outcome': [gen_labels[i] for i in range(len(gen_labels))]})
    return pd.concat([gen_data, gen_labels], axis=1)

def fit_synthcity(data, title):
    # generating synthetic data using synthcity
    X = data.iloc[:, :-1]
    X["target"] = data.iloc[:, -1]
    # A conditional generative adversarial network which can handle tabular data.
    syn_model = Plugins().get("ctgan")
    syn_model.fit(X)
    ctgan_data = handle_synthcitys_mess(syn_model, 700)
    ctgan_data.to_csv(f'datasets/ctgan_data_{title}.csv', index=False)

    # A conditional VAE network which can handle tabular data.
    syn_model = Plugins().get("tvae")
    syn_model.fit(X)
    tvae_data = handle_synthcitys_mess(syn_model, 700)
    tvae_data.to_csv(f'datasets/tvae_data_{title}.csv', index=False)


input_data = Support.read_file('datasets/diabetes.csv')
fit_synthcity(input_data, "indians")

input_data = Support.read_file('datasets/diabetic_mellitus.arff.csv')
fit_synthcity(input_data, "mellitus")