import stat

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from itertools import product

import sklearn as sk
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.svm import SVC
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import Birch

data_x_train = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\X_train.csv')
data_x_test = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\X_test.csv')
data_y_test = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\y_test_example.csv')
data_y_train = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\y_train.csv')

data_x_train['EU_Sales'] = pd.Series(data_y_train['EU_Sales'])

data_xy = data_x_train


# ---------functions----------#
def bars(a):
    plt.xlabel(a)
    plt.ylabel('Frequency')
    plt.title(a + ' Bars Plot')
    keys, counts = np.unique(data_x_train[a], return_counts=True)
    plt.bar(keys, counts, color="skyblue", edgecolor='blue', linewidth=1)
    plt.show()


def Change_Score_Value_Critic(val):
    return val / 100


def Change_Score_Value_User(val):
    return val / 10


def get_decay_from_year(year):
    if 1980 <= year < 1990:
        return 'A'
    elif 1990 <= year < 2000:
        return 'B'
    elif 2000 <= year < 2010:
        return 'C'
    else:
        return 'D'


def create_top_columns(data, column_name, top_num=10):
    new_column_name = 'is_top_' + column_name.lower()
    data[new_column_name] = 0
    top = (data[[column_name, 'Name']].groupby(column_name).count()).nlargest(top_num, 'Name')
    data.loc[data[column_name].isin(top.index), [new_column_name]] = 1
    return data


# ---------------------- convert to categories ----------------------#

# adding top developer columns based on num of sales
data_xy = create_top_columns(data_xy, 'Developer')
data_xy = create_top_columns(data_xy, 'Publisher', 5)
data_xy['Year_Categorical'] = data_xy['Year_of_Release'].apply(get_decay_from_year)
data_xy = data_xy.drop(columns=['Year_of_Release', 'Publisher', 'Developer'])  # removing old ones

# ---------------------- Normalizing ----------------------#
data_xy['Critic_Score_New'] = data_xy['Critic_Score'].apply(Change_Score_Value_Critic)
data_xy['User_Score_New'] = data_xy['User_Score'].apply(Change_Score_Value_User)
data_xy = data_xy.drop(columns=['Critic_Score', 'User_Score'])

# ---------------------- remove row ----------------------#
num_of_rows = data_xy.count()
data_xy = data_xy[data_xy.Name != "Wii Sports"]
data_xy = data_xy.drop(columns=['Name', 'Reviewed'])

data_xy = data_xy[data_xy['Rating'] != 'AO']
data_xy = data_xy[data_xy['Rating'] != 'RP']
data_xy = data_xy[data_xy['Rating'] != 'K-A']

# ---------------------- change y to 0 or 1 ----------------------#
data_xy_k_means = data_xy
sales = np.array(data_xy['EU_Sales']).reshape(-1, 1)
est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
est.fit(sales)
data_xy['y'] = est.transform(sales)
data_xy = data_xy.drop(columns=['EU_Sales'])

# add dummies variables

df_dummies = pd.get_dummies(data_xy,
                            columns=['Platform', 'Genre', 'Rating', 'is_top_publisher', 'is_top_developer',
                                     'Year_Categorical'], drop_first=False)
X = df_dummies.drop(columns=['y'])
y = df_dummies.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

print("Train\n-----------\n", pd.value_counts(y_train) / y_train.shape[0])
print("\nTest\n-----------\n", pd.value_counts(y_test) / y_test.shape[0])

# #---------------------------DT_Model--------------------------------------------------------------------------------

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)
#
# plt.figure(figsize=(12, 10))
# plot_tree(model, filled=True, class_names=['0', '1'])
# plt.show()

plt.figure(figsize=(12, 10))
plot_tree(model, filled=True, max_depth=2, class_names=['0', '1'], feature_names=X.columns)
plt.show()



export_graphviz(
    model,
    out_file="FullTree.dot",
    feature_names=X.columns,
    class_names=['0', '1'],
    filled=True
)

print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.3}")
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3}")

# tune max_depth
max_depth_list = np.arange(1, 50, 1)

res = pd.DataFrame()
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    res = res.append({'max_depth': max_depth,
                      'train_acc': accuracy_score(y_train, model.predict(X_train)),
                      'test_acc': accuracy_score(y_test, model.predict(X_test))}, ignore_index=True)

plt.figure(figsize=(13, 4))
plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
plt.plot(res['max_depth'], res['test_acc'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Test accuracy'])
plt.show()
print(res.sort_values('test_acc', ascending=False))
res.sort_values('test_acc', ascending=False)



# # # ------------------------ Train-Validation split ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

res = pd.DataFrame()
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    res = res.append({'max_depth': max_depth,
                      'train_acc': accuracy_score(y_train, model.predict(X_train)),
                      'val_acc': accuracy_score(y_val, model.predict(X_val))}, ignore_index=True)
plt.figure(figsize=(13, 4))
plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
plt.plot(res['max_depth'], res['val_acc'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()

best_max_depth = res.loc[res['val_acc'].idxmax(), 'max_depth']
print("Best value: ", best_max_depth)
model = DecisionTreeClassifier(criterion='entropy', max_depth=best_max_depth, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy: ", round(accuracy_score(y_val, model.predict(X_val)), 2))
# # -----------parameters tuning---------------

#Grid search
DecisionTreeClassifier()

param_grid = {'max_depth': np.arange(1, 25, 1),  # 101
              'criterion': ['entropy', 'gini'],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'min_samples_leaf': (1, 2, 3)
              }

comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print(comb)

print(param_grid['max_depth'])
print(param_grid.values())

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           refit=True,
                           cv=10)

grid_search.fit(X_train, y_train)
print("Grid search results: ")
best_model = grid_search.best_estimator_
print("preds: ")
preds = best_model.predict(X_test)
print("Test accuracy: ", round(accuracy_score(y_test, preds), 3))
print("preds2: ")
preds2 = best_model.predict(X_train)
print("Train Accuracy: ", round(accuracy_score(y_train, preds2), 3))
#
best_model.fit(X_train, y_train)

plt.figure(figsize=(12, 10))
plot_tree(best_model, filled=True, class_names=True)
plt.show()
#
print("Feature importances:")
print()
print(best_model.feature_importances_)

# # # ---DT chosen model----

print("for DT")
final_model = DecisionTreeClassifier(max_depth=13, criterion='entropy',
                                     max_features=None, min_samples_leaf=3, random_state=42)
final_model.fit(X_train, y_train)

export_graphviz(
    final_model,
    out_file=("FullTree.dot"),
    feature_names=X.columns,
    class_names=['0', '1'],
    filled=True
)

plt.figure(figsize=(12, 10))
plot_tree(final_model, max_depth=2, filled=True, class_names=True)
plt.show()

X_columns = list(X.columns)
print(X_columns)
print("Feature importances:")
print()
print(final_model.feature_importances_)
dict_feature = dict(zip(X_columns, list(final_model.feature_importances_)))
print(dict_feature)
import json
print(json.dumps(dict_feature, indent=4, sort_keys=True))


print(f"Train Accuracy: {accuracy_score(y_true=y_train, y_pred=final_model.predict(X_train)):.4f}")
print(f"Test Accuracy: {accuracy_score(y_true=y_test, y_pred=final_model.predict(X_test)):.4f}")

print(confusion_matrix(y_true=y_test, y_pred=final_model.predict(X_test)))

print('confusion matrix for DT:')
print(confusion_matrix(y_true=y_test, y_pred=final_model.predict(X_test)))
matrix_confusion = confusion_matrix(y_true=y_test, y_pred=final_model.predict(X_test))
precision = (matrix_confusion[0][0]) / ((matrix_confusion[0][0]) + (matrix_confusion[0][1]))
RECALL = (matrix_confusion[0][0]) / ((matrix_confusion[0][0]) + (matrix_confusion[1][0]))
F1 = 2 * (precision * RECALL) / (precision + RECALL)
print('precision for DT', precision)
print('RECALL for DT', RECALL)
print('F1 for DT', F1)

# -------------------------------- ANN  --------------------------------------------------

##First 2 models

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

modelThreeLayer = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(13,4),
                      max_iter=1000,
                      activation='logistic',
                      )
first =modelThreeLayer.fit(X_train_s, y_train)
print(first)

plt.plot(modelThreeLayer.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

print(f"Train accuracy: {sk.metrics.accuracy_score(y_train, first.predict(X_train_s)):.2}")
print(f"Validation accuracy: {sk.metrics.accuracy_score(y_test, first.predict(X_test_s)):.2}")


modelFourLayer = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(13,4),
                      max_iter=1000,
                      activation='logistic',
                      )
second =modelFourLayer.fit(X_train_s, y_train)
print(second)

plt.plot(modelFourLayer.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

print(f"Train accuracy: {sk.metrics.accuracy_score(y_train, second.predict(X_train_s)):.2}")
print(f"Validation accuracy: {sk.metrics.accuracy_score(y_test, second.predict(X_test_s)):.2}")



#####Random search - this is only one of the iterationd we manually did

randomValueRange = {'random_state': [0],
                        'hidden_layer_sizes': [x for x in itertools.product((np.arange(24, 100, 3)), repeat=2)],
                        'activation': ['logistic',],
                        'solver' :['sgd'],
                        'alpha' :[0.001 ,0.01],
                        'learning_rate_init' : [0.01 , 0.001 , 0.0001] ,
                        'max_iter' : np.arange(10,50 ,2),
                        'early_stopping' : [True,False]
                        }

comb = 1
for list_ in randomValueRange.values():
    comb *= len(list_)
    print(comb)


random_search = RandomizedSearchCV(MLPClassifier(random_state=1),
                                   param_distributions=randomValueRange,cv=10,
                                   random_state=1, n_iter=100, refit=True, return_train_score=True)

random_search.fit(X_test, y_test)
BestModelFound = random_search.best_estimator_



print(BestModelFound)
trainAccuracy = accuracy_score(y_true=y_train, y_pred=BestModelFound.predict(X_train_s))
testAccuracy = accuracy_score(y_true=y_test, y_pred=BestModelFound.predict(X_test))
print("The accuracy for train set is ",f"Accuracy: {accuracy_score(y_true=y_train, y_pred=BestModelFound.predict(X_train_s)):.3f}")
print("The accuracy for test set is " ,f"Accuracy: {accuracy_score(y_true=y_test, y_pred=BestModelFound.predict(X_test)):.3f}")


scores = pd.DataFrame(random_search.cv_results_)
scores.to_excel(r'C:\Users\Dell\Desktop\לימוד מכונה\Part B\random search.xlsx', index = False)

# #  ------ best From Random ------ #
scaler = StandardScaler()
bestFromRandom = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(58,82),
                      max_iter=46,
                      activation='logistic',
                      alpha=0.001,
                      learning_rate_init=0.01,
                      verbose=True,
                      )
best =bestFromRandom.fit(X_train_s, y_train)
print(best.predict_proba)
print(best)

plt.plot(bestFromRandom.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

print(f"Train accuracy: {sk.metrics.accuracy_score(y_train, best.predict(X_train_s)):.2}")
print(f"Validation accuracy: {sk.metrics.accuracy_score(y_test, best.predict(X_test_s)):.2}")

print('confusion matrix for ANN:')
print(confusion_matrix(y_true=y_test, y_pred=best.predict(X_test_s)))
matrix_confusion=confusion_matrix(y_true=y_test, y_pred=best.predict(X_test_s))
precision=(matrix_confusion[0][0])/((matrix_confusion[0][0])+(matrix_confusion[0][1]))
RECALL=(matrix_confusion[0][0])/((matrix_confusion[0][0])+(matrix_confusion[1][0]))

### Was done to bring most controversial 4 games

# a = bestFromRandom.predict_proba(X_test_s)
# print(a)
# aa = pd.DataFrame(a)



print('confusion matrix for ANN:')
print(confusion_matrix(y_true=y_test, y_pred=best.predict(X_test_s)))
matrix_confusion=confusion_matrix(y_true=y_test, y_pred=best.predict(X_test_s))
precision=(matrix_confusion[0][0])/((matrix_confusion[0][0])+(matrix_confusion[0][1]))
RECALL=(matrix_confusion[0][0])/((matrix_confusion[0][0])+(matrix_confusion[1][0]))
F1 = 2 * (precision * RECALL) / (precision + RECALL)
print('precision for ANN',precision)
print('RECALL for ANN',RECALL)
print('F1 for ANN',F1)


# # # --------- SVM-----------

#normalization
scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train)  # scaling train data for ANN model
X_test_svm = scaler_svm.fit_transform(X_test)  # scaling validation data for ANN model

# # print("svm:")
# SVMGrid= {'kernel': ['linear'], 'C': np.arange(1, 5, 0.5), 'gamma': ['scale']}
# grid_searchSVM = GridSearchCV(estimator=SVC(random_state=42, max_iter=600), param_grid=SVMGrid, refit=True, cv=10)
# grid_searchSVM.fit(X_train_svm, np.ravel(y_train))
# print("Best SVM Hyper Parameters: ", grid_searchSVM.best_params_)
# print("Best SVM Score: ", round(grid_searchSVM.best_score_,3))

bestModelSVM = SVC(C=3.5, gamma='scale', kernel='linear', random_state=42)
bestModelSVM.fit(X_train_svm, np.ravel(y_train))

# Evaluate the estimator
train_score = bestModelSVM.score(X_train_svm, y_train)
valid_score = bestModelSVM.score(X_test_svm, y_test)

print("SVM - train_score: ", train_score)
print("SVM - valid_score: ", valid_score)

print('confusionmatrix for svm:')
print(confusion_matrix(y_true=y_test, y_pred=bestModelSVM.predict(X_test_svm)))
matrix_confusion = confusion_matrix(y_true=y_test, y_pred=bestModelSVM.predict(X_test_svm))
precision = (matrix_confusion[0][0]) / ((matrix_confusion[0][0]) + (matrix_confusion[0][1]))
RECALL = (matrix_confusion[0][0]) / ((matrix_confusion[0][0]) + (matrix_confusion[1][0]))
F1 = 2 * (precision * RECALL) / (precision + RECALL)
print('precision for svm', precision)
print('RECALL for svm', RECALL)
print('F1 for svm', F1)

print('And for a linear kernel, our fitted model is a hyperplane (ω^[T] x+ b = 0),')
print('where ω is the vector of weights and b is the intercept.')

print('weights: ')
print(bestModelSVM.coef_)
print('Intercept: ')
print(bestModelSVM.intercept_)

# # # # --------- k-means -----------
#  normalization
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # scaling train data for ANN model
X_test_kmeans = scaler.fit_transform(X_test)  # scaling validation data for ANN model
X_s = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X_train_s)

#  transfer to pc1 & pc2
X_norm_pca = pca.transform(X_train_s)
X_norm_pca = pd.DataFrame(X_norm_pca, columns=['PC1', 'PC2'])
X_norm_pca['y'] = y
#
# # # # ---------------model------------

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train_s)
kmeans.cluster_centers_
kmeans.predict(X_train_s)

X_norm_pca['cluster'] = kmeans.predict(X_train_s)
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_norm_pca)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+',
            s=100, color='red')
plt.show()

x_test = np.linspace(-4, 9, 206)
y_test = np.linspace(-3, 17, 206)
predictions = pd.DataFrame()
for x in x_test:
    for y in y_test:
        pred = kmeans.predict(pca.inverse_transform(np.array([x, y])).reshape(-1, 48))[0]
        predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)

plt.scatter(x=predictions[predictions.y == 0]['X1'], y=predictions[predictions.y == 0]['X2'], c='powderblue')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y=predictions[predictions.y == 1]['X2'], c='pink')
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_norm_pca)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+',
            s=500, color='red')
plt.show()

X_norm_pca_3_c = X_norm_pca
X_norm_pca_3_c['correct'] = np.where((X_norm_pca['cluster'] == X_norm_pca['y']), 1, 0)
correct_pred = X_norm_pca_3_c['correct'].sum()
accuracy = correct_pred / 5524

print("k-means accuracy for 2 clusters:", accuracy)

# # # #---------------- K clusters ----------------

from sklearn.metrics import silhouette_score, davies_bouldin_score

dbi_list = []
sil_list = []

for n_clusters in range(2, 10, 1):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    assignment = kmeans.predict(X)

    sil = silhouette_score(X, assignment)
    dbi = davies_bouldin_score(X, assignment)

    dbi_list.append(dbi)
    sil_list.append(sil)

plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()
print('show')

# # #----------------------- 3 labels ------------------------------------------------------------------ # #  # # #

# creating new labels for 3 class
sales1 = np.array(data_xy_k_means['EU_Sales']).reshape(-1, 1)
est1 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
est1.fit(sales1)
data_xy_k_means['y'] = est1.transform(sales1)
data_xy_k_means = data_xy_k_means.drop(columns=['EU_Sales'])
data_xy_k_means_dummies = pd.get_dummies(data_xy_k_means,
                                         columns=['Platform', 'Genre', 'Rating', 'is_top_publisher', 'is_top_developer',
                                                  'Year_Categorical'], drop_first=False)
X_k = data_xy_k_means_dummies.drop(columns=['y'])
y_k_3 = data_xy_k_means_dummies.y

pca = PCA(n_components=2)
pca.fit(X_train_s)

#  transfer to pc1 & pc2
X_norm_pca_3 = pca.transform(X_train_s)
X_norm_pca_3 = pd.DataFrame(X_norm_pca_3, columns=['PC1', 'PC2'])
X_norm_pca_3['y'] = y_k_3

sns.scatterplot(x='PC1', y='PC2', hue='y', data=X_norm_pca_3)
plt.show()

# # # # ---------------model------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, max_iter=100, n_init=10, random_state=42)
kmeans.fit(X_train_s)
kmeans.cluster_centers_
kmeans.predict(X_train_s)
X_norm_pca_3['cluster'] = kmeans.predict(X_train_s)
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_norm_pca_3, palette='Accent')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+',
            s=100, color='red')
plt.show()
x_test = np.linspace(-5, 10, 100)
y_test = np.linspace(-5, 15, 100)
predictions = pd.DataFrame()
for x in x_test:
    for y in y_test:
        pred = kmeans.predict(pca.inverse_transform(np.array([x, y])).reshape(-1, 48))[0]
        predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)

plt.scatter(x=predictions[predictions.y == 0]['X1'], y=predictions[predictions.y == 0]['X2'], c='powderblue')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y=predictions[predictions.y == 1]['X2'], c='ivory')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y=predictions[predictions.y == 1]['X2'], c='ivory')

sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=X_norm_pca_3, palette='Accent')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+',
            s=100, color='red')
plt.show()

X_norm_pca_3_b = X_norm_pca_3
X_norm_pca_3_b['correct'] = np.where((X_norm_pca_3['cluster'] == X_norm_pca_3['y']), 1, 0)
correct_pred = X_norm_pca_3_b['correct'].sum()
accuracy = correct_pred / 5524

print("k-means accuracy for 3 clusters:", accuracy)

print('show')

# ------------------------------------------------------------ improvement DT -----------------------------------


X_new = data_x_train


def get_decay_from_year_new(year):
    if year < 2000:
        return 'A'
    elif 2000 <= year < 2010:
        return 'B'
    else:
        return 'C'


def split_genre_new(Genre):
    if Genre == 'action':
        return 'A'
    elif Genre == 'Racing':
        return 'B'
    elif Genre == 'Sports':
        return 'C'
    else:
        return 'D'


def replace_rating(Rating):
    if Rating in {'AO', 'RP', 'K-A'}:
        return 'other'
    else:
        return Rating


# ---------------------- convert to categories ----------------------#

# adding top developer columns based on num of sales
X_new = create_top_columns(X_new, 'Developer')
X_new = create_top_columns(X_new, 'Publisher', 5)
X_new['Year_Categorical'] = X_new['Year_of_Release'].apply(get_decay_from_year_new)
X_new['Genre_Categorical'] = X_new['Genre'].apply(split_genre_new)
X_new = X_new.drop(columns=['Genre', 'Year_of_Release', 'Publisher', 'Developer'])  # removing old ones

# ---------------------- Normalizing ----------------------#
X_new['Critic_Score_New'] = X_new['Critic_Score'].apply(Change_Score_Value_Critic)
X_new['User_Score_New'] = X_new['User_Score'].apply(Change_Score_Value_User)
X_new = X_new.drop(columns=['Critic_Score', 'User_Score'])

# ---------------------- remove row ----------------------#
num_of_rows = X_new.count()
X_new = X_new[X_new.Name != "Wii Sports"]
X_new = X_new.drop(columns=['Name', 'Reviewed'])

X_new['Rating'] = X_new['Rating'].apply(replace_rating)

# ---------------------- change y to 0 or 1 ----------------------#
sales = np.array(X_new['EU_Sales']).reshape(-1, 1)
est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
est.fit(sales)
X_new['y'] = est.transform(sales)
X_new = X_new.drop(columns=['EU_Sales'])

# add dummies variables

df_dummies = pd.get_dummies(X_new,
                            columns=['Platform', 'Rating', 'is_top_publisher', 'is_top_developer',
                                     'Year_Categorical', 'Genre_Categorical'], drop_first=False)
X = df_dummies.drop(columns=['y'])
y = df_dummies.y
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.1, random_state=4)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# Grid search
DecisionTreeClassifier()

param_grid = {'max_depth': np.arange(4, 20, 1),  # 101
              'criterion': ['entropy'],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'min_samples_leaf': (1, 2, 3, 4, 5, 6, 7, 8),
              'splitter': ['best', 'random']
              }

comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print(comb)

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           refit=True,
                           cv=5)

grid_search.fit(X_train_new, y_train_new)
print("Grid search results: ")
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test_new)
print("Test accuracy: ", round(accuracy_score(y_test_new, preds), 3))
preds2 = best_model.predict(X_train_new)
print("Train Accuracy: ", round(accuracy_score(y_train_new, preds2), 3))

# ------------------ final_model ---------------------------

final_model_new = DecisionTreeClassifier(max_depth=18, criterion='entropy',
                                         max_features=None, min_samples_leaf=3, splitter='best', random_state=42)
final_model_new.fit(X_train_new, y_train_new)

export_graphviz(
    final_model_new,
    out_file=("FullTree.dot"),
    feature_names=X.columns,
    class_names=['0', '1'],
    filled=True
)

X_columns = list(X.columns)
print(X_columns)
print("Feature importances:")
print()
print(final_model_new.feature_importances_)
dict_feature = dict(zip(X_columns, list(final_model_new.feature_importances_)))
print(dict_feature)
import json

print(json.dumps(dict_feature, indent=4, sort_keys=True))

print(f"Train Accuracy after improvement: {accuracy_score(y_true=y_train_new, y_pred=final_model_new.predict(X_train_new)):.3f}")
print(f"Test Accuracy after improvement: {accuracy_score(y_true=y_test_new, y_pred=final_model_new.predict(X_test_new)):.3f}")

#----------------------------------------------------------------------Predictions --------------------------------------------------------------------------------




# ---------functions----------#
def bars(a):
    plt.xlabel(a)
    plt.ylabel('Frequency')
    plt.title(a + ' Bars Plot')
    keys, counts = np.unique(data_x_train[a], return_counts=True)
    plt.bar(keys, counts, color="skyblue", edgecolor='blue', linewidth=1)
    plt.show()


def Change_Score_Value_Critic(val):
    return val / 100


def Change_Score_Value_User(val):
    return val / 10


def get_decay_from_year(year):
    if 1980 <= year < 1990:
        return 'A'
    elif 1990 <= year < 2000:
        return 'B'
    elif 2000 <= year < 2010:
        return 'C'
    else:
        return 'D'


def create_top_columns(data, column_name, top_num=10):
    new_column_name = 'is_top_' + column_name.lower()
    data[new_column_name] = 0
    top = (data[[column_name, 'Name']].groupby(column_name).count()).nlargest(top_num, 'Name')
    data.loc[data[column_name].isin(top.index), [new_column_name]] = 1
    return data


# ---------------------- convert to categories ----------------------#

# adding top developer columns based on num of sales
data_x_test = create_top_columns(data_x_test, 'Developer')
data_x_test = create_top_columns(data_x_test, 'Publisher', 5)
data_x_test['Year_Categorical'] = data_x_test['Year_of_Release'].apply(get_decay_from_year)
data_x_test = data_x_test.drop(columns=['Year_of_Release', 'Publisher', 'Developer'])  # removing old ones

# ---------------------- Normalizing ----------------------#
data_x_test['Critic_Score_New'] = data_x_test['Critic_Score'].apply(Change_Score_Value_Critic)
data_x_test['User_Score_New'] = data_x_test['User_Score'].apply(Change_Score_Value_User)
data_x_test = data_x_test.drop(columns=['Critic_Score', 'User_Score'])

# ---------------------- remove row ----------------------#
data_x_test = data_x_test[data_x_test.Name != "Wii Sports"]
data_x_test = data_x_test.drop(columns=['Name', 'Reviewed'])

data_x_test = data_x_test[data_x_test['Rating'] != 'AO']
data_x_test = data_x_test[data_x_test['Rating'] != 'RP']
data_x_test = data_x_test[data_x_test['Rating'] != 'K-A']

# --------------------add dummies variables ------------#

data_x_test_final = pd.get_dummies(data_x_test,columns=['Platform', 'Genre', 'Rating', 'is_top_publisher', 'is_top_developer','Year_Categorical'], drop_first=False)
data_x_test_final['Year_Categorical_A']=0
data_x_test_final=data_x_test_final[list(X_train.columns)]
# # ---DT chosen model----
#

project_final_model = DecisionTreeClassifier(max_depth=13, criterion='entropy',
                                     max_features=None, min_samples_leaf=3, random_state=42)

project_final_model.fit(X_train, y_train)

predictions =pd.DataFrame(project_final_model.predict(data_x_test_final))

print(predictions)
predictions.to_excel(r'C:\Users\karmo\PycharmProjects\ml11\predictions.xlsx',)
print('yess')
