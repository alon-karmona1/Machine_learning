import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from itertools import product

data_x_train = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\X_train.csv')
data_x_test = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\X_test.csv')
data_y_test = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\y_test_example.csv')
data_y_train = pd.read_csv(r'C:\Users\karmo\PycharmProjects\ml11\y_train.csv')


def print_hist(a):
    plt.xlabel(a)
    plt.ylabel('Frequency')
    plt.title(a + ' Histogram Plot')
    print('the max value of ' + a + ' is {}'.format(data_x_train[a].max()))
    print('the min value of ' + a + ' is {}'.format(data_x_train[a].min()))
    print('the mean value of ' + a + ' is {}'.format(data_x_train[a].mean()))
    print()
    plt.show()


def print_hist_EU_sales(a):
    plt.xlabel(a)
    plt.ylabel('Frequency')
    plt.title(a + ' Histogram Plot')
    print('the max value of' + str(a) + 'is {}'.format(data_y_train[a].max()))
    print('the min value of' + str(a) + 'is {}'.format(data_y_train[a].min()))
    print('the mean value of' + str(a) + 'is {}'.format(data_y_train[a].mean()))
    plt.show()


# histogram of continuous param

######################### EU_sales #############################
plt.hist(data_y_train['EU_Sales'], bins=50, range=(0,1), color="skyblue", edgecolor='blue', linewidth=1)
print_hist_EU_sales('EU_Sales')

plt.hist(data_x_train['User_Score'], bins=25, range=(0, 10), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('User_Score')

plt.hist(data_x_train['User_Count'], bins=50, range=(0,200), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('User_Count')

plt.hist(data_x_train['Critic_Count'], bins=25, range=(0,113), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('Critic_Count')

plt.hist(data_x_train['Critic_Score'], bins=25, range=(0,100), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('Critic_Score')

plt.hist(data_x_train['Critic_Score'], bins=25, range=(0,100), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('Critic_Score')

plt.hist(data_x_train['NA_Sales'], bins=50, range=(0,4), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('NA_Sales')

plt.hist(data_x_train['JP_Sales'], bins=50, range=(0,0.5), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('JP_Sales')

plt.hist(data_x_train['Other_Sales'], bins=50, range=(0,1), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('Other_Sales')

plt.hist(data_x_train['Year_of_Release'], bins=37, range=(1980,2016), color="skyblue", edgecolor='blue', linewidth=1)
print_hist('Year_of_Release')


# histogram of discrete param

def bars(a):
    plt.xlabel(a)
    plt.ylabel('Frequency')
    plt.title(a + ' Bars Plot')
    keys, counts = np.unique(data_x_train[a], return_counts=True)
    plt.bar(keys, counts, color="skyblue", edgecolor='blue', linewidth=1)
    plt.show()


bars('Genre')
bars('Rating')
bars('Developer')
bars('Platform')
bars('Publisher')


---------------------- continuous to categorized fichers----------------------#

def get_decay_from_year(year):
    if 1980 <= year < 1990:
        return 'A - 1980s'
    elif 1990 <= year < 200:
        return 'B - 1990s'
    elif 2000 <= year < 2010:
        return 'C - 2000s'
    else:
        return 'D - 2010s'




data_x_train['year_categorical'] = data_x_train['Year_of_Release'].apply(get_decay_from_year)
bars('year_categorical')

---------------------- BoxPlots----------------------#

sn.set_theme(style="whitegrid")
sn.boxplot(data=data_x_train, x = 'User_Score' , color ='skyblue')
plt.show()
sn.boxplot(data=data_x_train, x = 'User_Count', color ='skyblue')
plt.show()
sn.boxplot(data=data_x_train, x = 'Critic_Score' , color ='skyblue')
plt.show()
sn.boxplot(data=data_x_train, x = 'Critic_Count' , color ='skyblue')
plt.show()
sn.boxplot(data=data_x_train, x = 'NA_Sales' , color ='skyblue')
plt.show()
sn.boxplot(data=data_x_train, x = 'JP_Sales' , color ='skyblue')
plt.show()
sn.boxplot(data=data_x_train, x = 'Other_Sales' , color ='skyblue')
plt.show()
sn.boxplot(data=data_y_train, x = 'EU_Sales' , color ='skyblue')
plt.show()


---------------------- remove row ----------------------#
data_x_train.drop(data_x_train.index[2951])
data_x_train.drop[data_x_train.name = 'Wii Sports']
print(data_x_train.loc[2951])
print(data_x_train.drop(data_x_train.index[2951]))
print(data_x_train.loc[2951])

---------------------- correlation matrix ----------------------#
corr_matrix = data_x_train.corr()
sn.heatmap(corr_matrix, annot=True, cmap="Blues", annot_kws={"size": 30})
plt.show()

---------------------- count results ----------------------#
data_x_train[['Genre']].groupby('Genre').count()
a = (data_x_train[['Genre', 'Name']].groupby('Genre').count())/(6142)
# data_x_train[['Genre', 'User_Score']].groupby('Genre').sum()

---------------------- correlation_x_y ----------------------#
def corr_x_y(a):
    #     plt.xlabel(a)
    #     plt.ylabel('EU_Sales')
    #     plt.title(a + ' and EU_Sales Plot')
    #     plt.plot(data_x_train[a], data_y_train['EU_Sales'], '.', color='darkblue')
    #     plt.show()

for i in data_x_train.columns:
    corr_x_y(i)

---------------------- corr_x_y value ----------------------#
print('The corr between EU_Sales and Critic_Count is: ' + str(data_x_train['Critic_Count'].corr(data_y_train['EU_Sales'])))
print('The corr between EU_Sales and Critic_Score is: ' + str(data_x_train['Critic_Score'].corr(data_y_train['EU_Sales'])))
print('The corr between EU_Sales and User_Count is: ' + str(data_x_train['User_Count'].corr(data_y_train['EU_Sales'])))
print('The corr between EU_Sales and User_Score is: ' + str(data_x_train['User_Score'].corr(data_y_train['EU_Sales'])))
print('The corr between EU_Sales and NA_Sales is: ' + str(data_x_train['NA_Sales'].corr(data_y_train['EU_Sales'])))
print('The corr between EU_Sales and Other_Sales is: ' + str(data_x_train['Other_Sales'].corr(data_y_train['EU_Sales'])))
print('The corr between EU_Sales and JP_Sales is: ' + str(data_x_train['JP_Sales'].corr(data_y_train['EU_Sales'])))

---------------------- combine 2 categories & EU_Sales_heatmap ----------------------#
def Change_Values(val):
    if val > 1:
        return 1
    else:
        return val


data_y_train['EU_Sales_heatmap'] = data_y_train['EU_Sales'].apply(Change_Values)
plt.scatter(data_x_train['Critic_Count'], data_x_train["User_Score"], c=data_y_train['EU_Sales_heatmap'], s=4,
            marker='o')
plt.colorbar().set_label('EU_Sales_range', fontsize=14)
plt.xlabel('Critic_Count', fontsize=14)
plt.ylabel('User_Score', fontsize=14)
plt.title('EU_Sales bias Critic_Count & User_Score ', fontsize=17)
plt.show()


---------------------- Feature Representation: ----------------------#
def Change_Score_Value_Critic(val):
    return val / 100


def Change_Score_Value_User(val):
    return val / 10


data_x_train['Critic_Score_New'] = data_x_train['Critic_Score'].apply(Change_Score_Value_Critic)
data_x_train['User_Score_New'] = data_x_train['User_Score'].apply(Change_Score_Value_User)
data_x_train = data_x_train.drop(columns=['Critic_Score','User_Score'])
print()
---------------------- remove Exceptions  ----------------------#
data_x_train.drop([2951])

data_x_train = data_x_train[data_x_train.Name != "Wii Sports"]
data_x_train = data_x_train.drop(columns=['Name', 'Reviewed'])
print()

plt.plot(data_x_train['NA_Sales'], data_x_train['Other_Sales'], '.', color='darkblue')
plt.ylim((0,4))
plt.xlim((0,15))
plt.show()