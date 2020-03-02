import pandas as pd
from part2_claim_classifier import *
from part3_pricing_model import *
from part3_pricing_model import load_model as part3_loader

classifier = part3_loader()
data = pd.read_csv("part3_training_data.csv")
X = data.drop(columns=["claim_amount", "made_claim"])
res1 = classifier.predict_premium(X)
res2 = classifier.predict_claim_probability(X)
print("1: ", res1)
print("2: ", res2)