import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

hours = [1,2,3,4,5,6,7,8,9,10]
scores = [10,20,30,40,50,60,65,75,85,95]

data = pd.DataFrame({'Hours': hours, 'Scores': scores})

plt.scatter(data['Hours'], data['Scores'])
plt.title('Study Hours vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('score')
plt.show()

model = LogisticRegression()

model.fit(data[['Hours']], data['Scores'])

predicted = model.predict([[6.5]])

print(f"Predicted Score for 6.5 hours: {predicted[0]:.2f}")