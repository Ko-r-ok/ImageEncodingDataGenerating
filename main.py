import Support

support = Support.Support(batch_size = 16, n_classes = 2)
fake_samples = support.fit("datasets/diabetes.csv", ['BMI', 'DiabetesPedigreeFunction'], "Pima Indians")
fake_samples.to_csv(f'datasets/cgan_data_indians.csv', index=False)

fake_samples = support.fit("datasets/diabetic_mellitus.arff", ['BMI'], "Mellitus")
fake_samples.to_csv(f'datasets/cgan_data_mellitus.csv', index=False)