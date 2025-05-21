import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('wisconsin_breastcancer.csv')

df = df.dropna(axis=1)

X, y = df.drop('diagnosis', axis=1), df['diagnosis']

label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression()
lr_grid = GridSearchCV(lr, {'C': [0.59139, 0.5914, 0.59141]}, scoring='recall', error_score='raise')
lr_grid.fit(X_scaled, y)
print(lr_grid.best_estimator_)
print(lr_grid.best_score_)



