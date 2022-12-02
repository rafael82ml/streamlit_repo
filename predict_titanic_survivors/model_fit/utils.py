import pandas as pd
from sklearn.model_selection import train_test_split
import os
DIRECTORY = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(f'{DIRECTORY}/../data/train.csv', index_col=0)

y = df['Survived']
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
X_train, X_test, y_train, y_test = train_test_split(X,y)

# age_means = df.groupby(['Pclass','Sex'])['Age'].mean()
age_mean = X_train['Age'].mean()

