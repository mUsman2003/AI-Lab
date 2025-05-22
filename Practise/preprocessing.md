# 📘 Data Preprocessing with Python 
## Exploratory Data Analysis (EDA)
### Check for Missing Values in a Specific Column
```python
df['column_name'].isnull()              # Boolean Series for null values
df['column_name'].isnull().sum()        # Count of null values in the column
```
### Check for Duplicate Rows
```python
df.duplicated()                         # Boolean Series for duplicated rows
df['column_name'].duplicated().sum()    # Count duplicated values in a specific column
df.drop_duplicates(inplace=True)        # Drop duplicate rows
```
### Summary Statistics of a Specific Column
```python
df['column_name'].describe()            # Statistical summary (count, mean, std, etc.)
df['column_name'].value_counts()        # Count of unique values
```
### Correlation Involving Specific Columns
```python
df[['col1', 'col2']].corr()             # Correlation between two specific columns
```

## Handling Missing Values
### Remove Rows with Missing Values in a Specific Column
```python
df.dropna(subset=['column_name'], inplace=True)
```
### Fill Missing Values in a Specific Column
```python
df['column_name'].fillna(df['column_name'].mean(), inplace=True)        # Fill with Mean 
df['column_name'].fillna(df['column_name'].median(), inplace=True)      # Fill with Median
df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)     # Fill with Mode
df['column_name'].fillna(0, inplace=True)                               # Replace 0 with any desired constant
```

## Feature Scaling
### Standardization using StandardScaler on Specific Columns
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['col1', 'col2']])  # Only selected columns
```

## Splitting Data into Train and Test Sets

### Code Example:
```python
from sklearn.model_selection import train_test_split

# Assume X contains features and y contains the target variable
X = df.drop('target_column', axis=1)  # Features
y = df['target_column']               # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Confusion Matrix

**Definition:**  
A confusion matrix is a table used to describe the performance of a classification model by showing the true vs. predicted values.

|                    | Predicted: No (0) | Predicted: Yes (1) |
|--------------------|------------------|---------------------|
| Actual: No (0)     | True Negative (TN) | False Positive (FP) |
| Actual: Yes (1)    | False Negative (FN) | True Positive (TP)  |

## Accuracy

**Definition:**  
Accuracy measures how many total predictions (both positives and negatives) were correct out of all predictions made.

**Formula:**  
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

## Precision

**Definition:**  
Precision measures how many of the predicted positive values are actually positive.

**Formula:**  
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

## Recall (Sensitivity or True Positive Rate)

**Definition:**  
Recall measures how many of the actual positive values were correctly predicted by the model.

**Formula:**  
$$
\text{Recall} = \frac{TP}{TP + FN}
$$


## F1 Score

**Definition:**  
The F1 Score is the harmonic mean of precision and recall. It balances the trade-off between the two.

**Formula:**  
$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

## Built in Fuction
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report
acc = accuracy_score(y_test, y_pred)            
```

## Classes
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hi, I'm {self.name} and I'm {self.age} years old.")
```
## Sorting
```python
nums = [5, 2, 9, 1]
nums.sort()                 # In-place sort
sorted_nums = sorted(nums)  # Returns a new sorted list
words = ["apple", "banana", "cherry"]
sorted_by_length = sorted(words, key=len)
nums.sort(reverse=True)     # Reverse sort
```
# Type Conversion
```python
int("10")        # 10
float("3.14")    # 3.14
str(123)         # '123'
list("abc")      # ['a', 'b', 'c']
set([1, 2, 2])   # {1, 2}
dict([("a", 1)]) # {'a': 1}
```
## String Manipulation
```python
s = "hello world"
s.upper()         # 'HELLO WORLD'
s.lower()         # 'hello world'
s.title()         # 'Hello World'
s.strip()         # removes leading/trailing whitespace
s.replace("l", "*") # 'he**o wor*d'
s.split()         # ['hello', 'world']
",".join(['a', 'b']) # 'a,b'
```
## List Operations
```python
lst = [1, 2, 3]
lst.append(4)
lst.insert(1, 10) # [1, 10, 2, 3, 4]
lst.remove(2)     # removes first 2
lst.pop()         # removes last item
lst[1:3]          # slicing
len(lst)
sum(lst)
```
## Dictionary Operations
```python
d = {"a": 1, "b": 2}
d["c"] = 3
val = d.get("a", 0)
for key, val in d.items():
    print(key, val)
del d["b"]
"d" in d          # check key
```
## Set Operations
```python
s1 = {1, 2, 3}
s2 = {2, 3, 4}

s1.add(4)
s1.remove(1)
s1.union(s2)      # {1, 2, 3, 4}
s1.intersection(s2) # {2, 3}
s1.difference(s2) # {1}
```
## Array Operations (NumPy)
```python
import numpy as np

arr = np.array([1, 2, 3])
arr.shape
arr.reshape((3, 1))
arr + 1
arr.mean()
arr.sum()
arr[1:3]
```
```python
squares = [x*x for x in range(5)]
evens = [x for x in range(10) if x % 2 == 0]
```