import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import Datasets + Pre-processing


# Exploratory Data Analysis
# Find means, medians, outliers, etc. on all the datasets


# Search for all possible outliers and missing values within the datasets



# Plot out the labels for each borough within the stated time span



# Perform a feature breakdown on the features imported

# Perform a correlation heatmap? to find to what degree which features correlate with other features


# Check for feature variance, and fit accordingly, so that features with higher variance do not have a higher pull in the prediction.
# In other words, normalize the data



# Engineer any potential features that may be useful

# Feature importance preview with Random Forest (for selection rationale)


# PCA & t-SNE Visualization (optional)



# Training, Testing, and Splitting




# Hyperparameter Tuning


# Final Evaluation