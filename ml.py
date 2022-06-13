import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Load the dataset
dataset = pd.read_csv("data/diabetes.csv")

print(dataset.head())

# Check if there are null values
print(dataset.isnull().any())
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
print(dataset.corr())

# Visualize the correlation of Age and Pregnancies (54% correlated)
sns.jointplot(x='Age', y='Pregnancies', data=dataset)
plt.title("Correlation of Age and Pregnancies")
plt.show()

# Visualize the distribution of patients' ages
sns.countplot(x='Age', data=dataset)
plt.title("Age Distribution of Patients")
plt.show()

# Visualize the correlation of Outcome and Glucose (46% correlated)
sns.boxplot(x='Outcome', y='Glucose', data=dataset)
plt.title("Correlation of Outcome and Glucose")
plt.show()

# Count the number of Outcome (0:1 = 500:267)
sns.countplot(x='Outcome', data=dataset)
plt.title("Count of Outcome")
plt.show()
# Since the Outcome data is unbalanced (biased with negative outcome),
# rebalancing is needed

# Create two different dataframe of majority and minority class
df_majority = dataset[(dataset['Outcome'] == 0)]
df_minority = dataset[(dataset['Outcome'] == 1)]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=len(df_majority),  # to match majority class
                                 random_state=42)  # reproducible results
# Concatenate majority class with upsampled minority class
df_upsampled: DataFrame | Series = pd.concat([df_minority_upsampled, df_majority])

# Plot countplot to verify level of each class
sns.countplot(x='Outcome', data=df_upsampled)
plt.title("Rebalanced Outcome")
plt.show()

# Train the model with the data
X = df_upsampled.drop('Outcome', axis=1)
y = df_upsampled['Outcome']
X_train, X_test, y_train, y_test = train_test_split(df_upsampled.drop('Outcome', axis=1),
                                                    df_upsampled['Outcome'], test_size=0.15,
                                                    random_state=101)

# Initialize the model with the chosen parameters
logmodel = LogisticRegression(max_iter=500, verbose=1)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

# Count true negative, false positive, false negative, true positive
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print("Confusion Matrix:")
print('TN: ', tn)
print('FP: ', fp)
print('FN: ', fn)
print('TP: ', tp)
# print(confusion_matrix(y_test, predictions))

# sns.heatmap(confusion_matrix(y_test, predictions), annot=True)
plot_confusion_matrix(logmodel, X_test, y_test)
plt.title('Confusion matrix - 15% test')
plt.show()

# Drop DiabetesPedigreeFunction column to see if this data is bias for the outcome
df_upsampled.drop(['DiabetesPedigreeFunction'], axis=1, inplace=True)

print("\nDataset without DiabetesPedigreeFunction:\n", df_upsampled.head())

# Again, train the model with the data
X = df_upsampled.drop('Outcome', axis=1)
y = df_upsampled['Outcome']

X_train, X_test, y_train, y_test = train_test_split(df_upsampled.drop('Outcome', axis=1),
                                                    df_upsampled['Outcome'], test_size=0.15,
                                                    random_state=101)

logmodel = LogisticRegression(max_iter=500, verbose=1)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

plot_confusion_matrix(logmodel, X_test, y_test)
plt.title('Confusion matrix - 15% test - No DiabetesPedigreeFunction')
plt.show()