"""Generates the final dataset all.csv from given training CSV."""

import pandas as pd
import sys
import math

if len(sys.argv) > 1:
    data = pd.read_csv(sys.argv[1])
else:
    print("You'll need a copy of the CheXpert dataset!\nYou can get a download link by filling out the form at the bottom of this page.\n\nhttps://stanfordmlgroup.github.io/competitions/chexpert/")
    exit(0)

frontal = data[data["Frontal/Lateral"] == "Frontal"]

frontal_pos = frontal[frontal["Lung Lesion"] == 1.0].sample(frac=1)
frontal_neg = frontal[frontal["Lung Lesion"] == -1.0].sample(frac=1)
frontal_neu = frontal[frontal["Lung Lesion"].isnull()].sample(frac=1)

pd.concat(
    [frontal_pos,
     frontal_neg,
     frontal_neu.head(
         frontal_pos.shape[0]-frontal_neg.shape[0]
     )]
).to_csv("all.csv")
