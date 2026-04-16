import pandas as pandas
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.metrics import
confusion_matrix, classification_report, roc_curve, auc
df=pd.read_csv("titanic.csv")
sns.countplot(x='Survived', data=df)
plt.title("Survival count")
plt.show()
sns.countplot(x='Sex', hue='Survived',data=df)
plt.title("Survival Count")
plt.show()
sns.countplot(x='Sex', hue='Survived',data=df)
plt.title("Survival by Gender")
plt.show()
sns.histplot(df['Age'].dropna(), bins=30,kde=True)
plt.title("Age Distribution")
plt.show()
cm= confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test,y_pred))
fpr, tpr, thresholds = roc_curve(y_test,y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %0. 2f" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Position Rate")
plt.ylabel("True Position Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
importances = model.feature_importances_features = X.coulmns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
plt.bar(['Accuracy'],[acc])
plt.ylim(0,1)
plt.title("Model Accuracy")
plt.show()