import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.tree import export_graphviz
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns



excelFile='G:/uni/DataMining/pythoncode/TravelInsurancePrediction.xlsx'
df=pd.read_excel(excelFile)

columnName1='Index'
columnName2='Age'
columnName3='EmploymentType'
columnName4='GraduateOrNot'
columnName5='AnnualIncome'
columnName6='FamilyMembers'
columnName7='ChronicDiseases'
columnName8='FrequentFlyer'
columnName9='EverTravelledAbroad'
columnName10='TravelInsurance'

medianIndex=df['Index'].median()
print("medianIndex:",medianIndex)
df=df.drop("Index",axis=1)

maxAge=df[columnName2].max()
minAge=df[columnName2].min()
medianAge=df[columnName2].median()
print("max Age:",maxAge)
print("min Age:",minAge)
print("median Age:", medianAge)
AgeArray=df[columnName2].values.reshape(-1,1)
scaler=MinMaxScaler(feature_range=(0,1))
NormalizeAge=scaler.fit_transform(AgeArray)
NormalizeAge_df=pd.DataFrame(NormalizeAge,columns=['NormalizeAge'])
df=pd.concat([df,NormalizeAge_df],axis=1)

UniqueNamesForJob=df[columnName3].unique()
NumberOfUniqueNamesForJob=len(UniqueNamesForJob)
print("unique names for job:",UniqueNamesForJob)
print("number of unique names for job:",NumberOfUniqueNamesForJob)
GovernmentSectorCheck='Government Sector'
CountGovernmentSector=df[columnName3].value_counts()[GovernmentSectorCheck]
print("times that Government Sector is job:",CountGovernmentSector)
PrivateSectorCheck='Private Sector/Self Employed'
CountPrivateSector=df[columnName3].value_counts()[PrivateSectorCheck]
print("times that Private Sector is job:",CountPrivateSector)

CheckGraduated='Yes'
CountOfGraduatedPeople=df[columnName4].value_counts()[CheckGraduated]
print("number of graduated people:",CountOfGraduatedPeople)
CheckDontGraduated='No'
CountOfDontGraduatedPeople=df[columnName4].value_counts()[CheckDontGraduated]
print("number of dont graduated people:",CountOfDontGraduatedPeople)

maxAnnuallncome=df[columnName5].max()
minAnnuallncome=df[columnName5].min()
medianAnnuallncome=df[columnName5].median()
print("max Annuallncome:",maxAnnuallncome)
print("min Annuallncome:",minAnnuallncome)
print("median Annuallncome:",medianAnnuallncome)

maxFamilyMembers=df[columnName6].max()
minFamilyMembers=df[columnName6].min()
medianFamilyMembers=df[columnName6].median()
FamilyMembersArray=df[columnName6].values.reshape(-1,1)
scaler=MinMaxScaler(feature_range=(0,1))
NormalizeFamilyMember=scaler.fit_transform(FamilyMembersArray)
NormalizeFamilyMember_df=pd.DataFrame(NormalizeFamilyMember,columns=['NormalizeFamilyMember'])
df=pd.concat([df,NormalizeFamilyMember_df],axis=1)
print("max FamilyMembers:",maxFamilyMembers)
print("min FamilyMembers:",minFamilyMembers)
print("median FamilyMembers:",medianFamilyMembers)

OneInChronicDiseases=df[columnName7].value_counts()[1]
ZeroInChronicDiseases=df[columnName7].value_counts()[0]
print("number of ones in ChronicDiseases:", OneInChronicDiseases)
print("number of zeros in ChronicDiseases:",ZeroInChronicDiseases)

CheckHaveFrequentFlyer='Yes'
CountOfFrequentFlyerPeople=df[columnName8].value_counts()[CheckHaveFrequentFlyer]
print("number of people that have FrequentFlyer:",CountOfFrequentFlyerPeople)
CheckDontHaveFrequentFlyer='No'
CountOfDontFrequentFlyerPeople=df[columnName8].value_counts()[CheckDontHaveFrequentFlyer]
print("number of people thet dont have FrequentFlyer:",CountOfDontFrequentFlyerPeople)

CheckEverTravelledAbroad='Yes'
CountOfEverTravelledAbroadPeople=df[columnName9].value_counts()[CheckEverTravelledAbroad]
print("number of people thet EverTravelledAbroad:",CountOfEverTravelledAbroadPeople)
CheckDontTravelledAbroad='No'
CountOfDontTravelledAbroadPeople=df[columnName9].value_counts()[CheckDontTravelledAbroad]
print("number of people that NeverTravelledAbroad:",CountOfDontTravelledAbroadPeople)

OneInTravelInsurance=df[columnName10].value_counts()[1]
ZeroInTravelInsurance=df[columnName10].value_counts()[0]
print("number of ones in TravelInsurance:", OneInTravelInsurance)
print("number of zeros in TravelInsurance:",ZeroInTravelInsurance)


encoded_df=pd.get_dummies(df['EmploymentType'],prefix='EmploymentType')
df=pd.concat([df,encoded_df],axis=1)
df=df.drop('EmploymentType',axis=1)
df[columnName4]=df[columnName4].replace({'Yes':1,'No':0})
df[columnName8]=df[columnName8].replace({'Yes':1,'No':0})
df[columnName9]=df[columnName9].replace({'Yes':1,'No':0})
# df=df.drop([columnName3],axis=1)
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
# plt.title('heatmap of TravelInsurancePrediction')
# plt.show()


AgeCount=df[columnName2].value_counts().sort_index()
# plt.bar(AgeCount.index,AgeCount.values)
# plt.xlabel('Age')
# plt.ylabel('count')
# plt.title('Age Distribution')
# plt.show()


employment_counts = df[['EmploymentType_Government Sector', 'EmploymentType_Private Sector/Self Employed']].sum()
# plt.bar(employment_counts.index,employment_counts.values)
# plt.xlabel('Employment Type')
# plt.ylabel('count')
# plt.title('Employment Type Distribution')
# plt.show()

GraduateOrNotCount=df[columnName4].value_counts()
# plt.bar(GraduateOrNotCount.index,GraduateOrNotCount.values)
# plt.xlabel('GraduateOrNot')
# plt.ylabel('count')
# plt.title('GraduateOrNot Distribution')
# plt.show()

df['LogAnnualIncome']=np.log10(df['AnnualIncome'])
# plt.bar(df['LogAnnualIncome'].value_counts().index,df['LogAnnualIncome'].value_counts().values)
# plt.xlabel('AnnualIncome')
# plt.ylabel('count')
# plt.title('AnnualIncome Distribution')
# plt.show()

FamilyMembersCount=df[columnName6].value_counts().sort_index()
# plt.bar(FamilyMembersCount.index,FamilyMembersCount.values)
# plt.xlabel('FamilyMembers')
# plt.ylabel('count')
# plt.title('FamilyMembers Distribution')
# plt.show()

ChronicDiseasesCount=df[columnName7].value_counts()
# plt.bar(ChronicDiseasesCount.index,ChronicDiseasesCount.values)
# plt.xlabel('ChronicDiseases')
# plt.ylabel('count')
# plt.title('ChronicDiseases Distribution')
# plt.show()

FrequentFlyerCount=df[columnName8].value_counts()
# plt.bar(FrequentFlyerCount.index,FrequentFlyerCount.values)
# plt.xlabel('FrequentFlyer')
# plt.ylabel('count')
# plt.title('FrequentFlyer Distribution')
# plt.show()

EverTravelledAbroadCount=df[columnName9].value_counts()
# plt.bar(EverTravelledAbroadCount.index,EverTravelledAbroadCount.values)
# plt.xlabel('EverTravelledAbroad')
# plt.ylabel('count')
# plt.title('EverTravelledAbroad Distribution')
# plt.show()

TravelInsuranceCount=df[columnName10].value_counts()
# plt.bar(TravelInsuranceCount.index,TravelInsuranceCount.values)
# plt.xlabel('TravelInsurance')
# plt.ylabel('count')
# plt.title('TravelInsurance Distribution')
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(df.drop('TravelInsurance', axis=1), df['TravelInsurance'], test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Decision_Tree Gini:", accuracy)
dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns, class_names=['0', '1'], filled=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_Gini")
graph.view()


features = df.drop('TravelInsurance', axis=1)
labels = df['TravelInsurance']
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(train_data, train_labels)
predictions = model.predict(test_data)
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy Decision_Tree Entropy: ", accuracy)
dot_data = export_graphviz(model, out_file=None, feature_names=features.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_Entropy")
graph.view()


model_miss = DecisionTreeClassifier(criterion='gini')
model_miss.fit(train_data, train_labels)
predictions_miss = model_miss.predict(test_data)
accuracy_miss = accuracy_score(test_labels, predictions_miss)
print("Accuracy Decision_Tree Miss Classification: ", accuracy_miss)
dot_data_miss = export_graphviz(model_miss, out_file=None, feature_names=features.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
graph_miss = graphviz.Source(dot_data_miss)
graph_miss.render("decision_tree_Miss_Classification")
graph_miss.view()


svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Accuracy SVM with linear kernel:", accuracy_linear)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
rcv = RandomizedSearchCV(SVC(), param_grid, cv=5)
rcv.fit(X_train, y_train)
y_pred_svc = rcv.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("Accuracy SVC with best parameters:", accuracy_svc)
confusion_svc = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_svc, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(y_test, y_pred_svc))
print(f'\nBest Parameters of SVC model is: {rcv.best_params_}\n')


svm_poly = SVC(kernel='poly')
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print("Accuracy SVM with Polynomial kernel:", accuracy_poly)
param_grid_poly = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'degree': [2, 3, 4, 5]}
rcv_poly = RandomizedSearchCV(SVC(kernel='poly'), param_grid_poly, cv=5)
rcv_poly.fit(X_train, y_train)
y_pred_poly_best = rcv_poly.predict(X_test)
accuracy_poly_best = accuracy_score(y_test, y_pred_poly_best)
print("Accuracy SVM with best parameters (Polynomial kernel):", accuracy_poly_best)
confusion_poly = confusion_matrix(y_test, y_pred_poly_best)
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_poly, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(y_test, y_pred_poly_best))
print(f'\nBest Parameters of SVC model with Polynomial kernel is: {rcv_poly.best_params_}\n')



svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("Accuracy SVM with RBF kernel:", accuracy_rbf)
param_grid_rbf = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
rcv_rbf = RandomizedSearchCV(SVC(kernel='rbf'), param_grid_rbf, cv=5)
rcv_rbf.fit(X_train, y_train)
y_pred_rbf_best = rcv_rbf.predict(X_test)
accuracy_rbf_best = accuracy_score(y_test, y_pred_rbf_best)
print("Accuracy SVM with best parameters (RBF kernel):", accuracy_rbf_best)
confusion_rbf = confusion_matrix(y_test, y_pred_rbf_best)
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_rbf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(y_test, y_pred_rbf_best))
print(f'\nBest Parameters of SVC model with RBF kernel is: {rcv_rbf.best_params_}\n')


svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Accuracy SVM with linear kernel:", accuracy_linear)



svm_poly = svm.SVC(kernel='poly')
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print("Accuracy SVM with polynomial kernel:", accuracy_poly)


svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("Accuracy SVM with RBF kernel:", accuracy_rbf)



