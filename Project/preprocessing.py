import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

threthold = 10

def preprocessing(ncomp): 
    # Read data
    dataset = pd.read_csv('data_letter.csv')
    
    for col in dataset.columns: 
        if dataset[col].dtype == 'float64':
            mean = dataset[col].mean()
            dataset.fillna(mean, inplace=True)

    # Drop invalid and inefficient data
    dd_data = np.where(dataset['status'] == 'dd')[0]
    dataset = dataset.drop(dataset.index[dd_data])

    dd_data = np.where(dataset['status'] == 'Remove')[0]
    dataset = dataset.drop(dataset.index[dd_data])

    for i in np.unique(dataset['status']): 
        data_idx = np.where(dataset['status'] == i)[0]
        if len(data_idx) <= threthold: 
            dataset = dataset.drop(dataset.index[data_idx])

    status = dataset['status'].values
    _, y = np.unique(status, return_inverse=True)
    dataset = dataset.drop(['animal_name', 'status'], axis=1)

    # print(dataset.shape)
    for col in dataset.columns: 
        if dataset[col].dtype == object: 
            one_hot = pd.get_dummies(dataset[col])
            dataset = dataset.join(one_hot)
            dataset = dataset.drop([col], axis=1)

    x = dataset.values

    # y_value = list(np.unique(y))
    # count = np.zeros(len(y_value))
    # for i, y_val in enumerate(y): 
    #     y[i] = y_value.index(y_val)
    #     count[y_value.index(y_val)] += 1
    # print(count)
    # remove_2 = np.where(y == 0)[0]
    # remove_2 = remove_2[:800]
    # x = np.delete(x, remove_2, axis=0)
    # y = np.delete(y, remove_2)
    # print(np.where(y == 0)[0].shape)
    # divide the data into 70% and 30%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Perform PCA
    pca = PCA(n_components=ncomp)
    pca.fit(x_train, y_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    # sv = pca.singular_values_
    # PoV_len = sv.shape[0]
    # PoV = np.zeros(PoV_len)
    # for n in range(PoV_len): 
    #     PoV[n] = np.sum(sv[:n + 1] ** 2) / np.sum(sv[:ncomp] ** 2)
    # print('PoV: ')
    # print(PoV)

    return (x_train, x_test, y_train, y_test)