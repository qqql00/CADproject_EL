# ANOVA feature selection for numeric input and categorical output
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np

# load orignal dataset (features & labels)
# features = pd.read_csv("purefeatures_BM.csv", header=None)
# features = pd.read_csv("features_ResNet50.csv")
# features =pd.read_csv("UNET_Combi.csv")
features = pd.read_csv("features_CUtest1.csv")
labels = pd.read_csv("label_ResNet50.csv")
X = features.values
y = labels.values
print(np.shape(X))
print(np.shape(y))

# apply feature selection
# X_test_sel = SelectKBest(score_func=f_classif, k=100).transform(X)
X_selected = SelectKBest(score_func=f_classif, k=10).fit_transform(X, y)
print(X_selected.shape)
# save the selected features to .csv file
df = pd.DataFrame(X_selected)
df.to_csv("TCU_10.csv", index= False)
# df1 = pd.DataFrame(X_test_sel)
# df1.to_csv("test_selected.csv", index=False)