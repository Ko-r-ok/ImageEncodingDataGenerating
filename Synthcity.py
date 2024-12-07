from synthcity.plugins import Plugins
import pandas as pd

data = pd.read_csv('datasets/diabetes.csv')

# generate data with synthcity and transform it to a normal dataframe -- this is a mess
def handle_synthcitys_mess(model, amount):
    gen_data = model.generate(count=amount).unpack()
    gen_labels = gen_data[1]
    gen_data = gen_data[0]
    gen_labels = pd.DataFrame({'Outcome': [gen_labels[i] for i in range(len(gen_labels))]})
    return pd.concat([gen_data, gen_labels], axis=1)


# generating synthetic data using synthcity
X = data.iloc[:, :-1]
X["target"] = data.iloc[:, -1]

# A conditional generative adversarial network which can handle tabular data.
syn_model = Plugins().get("ctgan")
syn_model.fit(X)
ctgan_data = handle_synthcitys_mess(syn_model, 700)
# TODO unpack it to a dataframe
ctgan_data.to_csv('datasets/ctgan_data.csv', index=False)

# A conditional VAE network which can handle tabular data.
syn_model = Plugins().get("tvae")
syn_model.fit(X)
tvae_data = handle_synthcitys_mess(syn_model, 700)
tvae_data.to_csv('datasets/tvae_data.csv', index=False)