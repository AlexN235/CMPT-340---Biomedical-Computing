import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# read data
ecg_data = pd.read_csv('ecg_data.csv', header=None)
target = pd.DataFrame(np.zeros(4800))
target.iloc[2301:] = 1

# train test split
X_train, X_test, y_train, y_test = train_test_split(ecg_data,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=42)

# make PCA model with standard scaler
variance_propotion = 0.95
pca = make_pipeline(StandardScaler(), PCA(variance_propotion))
pca.fit(X_train)

# apply the mapping
X_train_PC = pca.transform(X_train)
X_test_PC = pca.transform(X_test)
print('From PCA %2d%% of total variance is explained by the first %3d PCs.' %
      (variance_propotion * 100, X_train_PC.shape[1]))

# build and fit a SVC model
print('A SVM classifier on original data')
model = SVC(C=10, kernel='rbf')
model.fit(X_train, y_train.values.ravel())
ori_score = model.score(X_test, y_test)
print('Test score: %.4f' % (ori_score))
print('A SVM classifier on PCA transformed data')
model_PC = SVC(C=10, kernel='rbf')
model_PC.fit(X_train_PC, y_train.values.ravel())
pc_score = model_PC.score(X_test_PC, y_test)
print('Test score: %.4f' % (pc_score))
