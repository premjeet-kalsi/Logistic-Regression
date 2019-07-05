# Logistic-Regression using Titanic dataset

## Observations and actions taken:
### Missing Values
The dataset had missing values for Age and Cabin column, after visualizing box plot for Pclass with age in the y axis, we came to know that wealthier passengers in higher class tend to be older. So made a function to fill the missing age values as mean value of the Pclass age.
```
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37   

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
```        
```
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
```
### Converting categorical features
Converted categorical features to dummy variables using pandas. Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.Sex and Embark columns have categorical data hence converting them to categorical features as follows:
```
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
```
