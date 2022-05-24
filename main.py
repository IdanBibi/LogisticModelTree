import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LogisticModelTree import LogisticModelTree
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import numpy as np


def analyze_data(data):
    categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # 1
    distribution = data['target'].value_counts()
    print("distribution of the labels: \n", distribution)
    fig1, ax1 = plt.subplots()
    ax1.pie(distribution.values, labels=distribution.keys(), autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Distribution of the labels:")
    plt.show()

    # 2
    print("NA values: \n", data.isna().sum(), "\n")

    # 3
    print(f"Statistics table: \n{data[[c for c in data.columns if c not in categorical and c != 'target']].describe()}")

    # 4
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Histograms of the features")
    data.hist(ax=ax)
    plt.show()

    # 5
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Correlation matrix of the features")
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    plt.show()


def preprocess(data):
    # 1
    categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    data = data.apply(lambda x: x.fillna(x.mean()) if type(x) in categorical else x.fillna(x.value_counts().index[0]),
                      axis=0)

    # 2
    y = data['target']
    X = data.drop('target', axis=1)

    # 3
    scaler = StandardScaler()
    columns = X.columns
    index = X.index
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=index, columns=columns)

    return X_scaled, y


def show_roc_curve(y_true, preds):
    pass


def task_4_configurations(X, y):
    model = LogisticModelTree()
    kf = KFold(n_splits=10, shuffle=True)
    val = 3

    for num in range(0, 3):
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train, test in kf.split(X, y):
            prediction = model.fit(X.iloc[train], y.iloc[train], min_leaf=5 - num, max_depth=val).predict(X.iloc[test])
            fpr, tpr, _ = roc_curve(y[test], prediction)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.2f )' % mean_auc, lw=2, alpha=1)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for model {num + 1}\nwith parameters: min_leaf={5 - num}, max_depth={val}')
        plt.legend(loc="lower right")
        plt.show()
        val += 1


# bonus
def calculate_class_weights(y):
    return {0: sum(y)/(len(y) - sum(y)), 1: (len(y) - sum(y))/sum(y)}

if __name__ == "__main__":
    data = pd.read_csv('heart.csv')
    #analyze_data(data)
    x, y = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=11)
    #weights = calculate_class_weights(y)

    # implement here the experiments for task 4
    task_4_configurations(x, y)
