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
import sklearn.cluster as cluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc





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
# graph.view()


features = df.drop('TravelInsurance', axis=1)
labels = df['TravelInsurance']
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
model_e = DecisionTreeClassifier(criterion='entropy')
model_e.fit(train_data, train_labels)
predictions = model_e.predict(test_data)
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy Decision_Tree Entropy: ", accuracy)
dot_data = export_graphviz(model_e, out_file=None, feature_names=features.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_Entropy")
# graph.view()


model_miss = DecisionTreeClassifier(criterion='gini')
model_miss.fit(train_data, train_labels)
predictions_miss = model_miss.predict(test_data)
accuracy_miss = accuracy_score(test_labels, predictions_miss)
print("Accuracy Decision_Tree Miss Classification: ", accuracy_miss)
dot_data_miss = export_graphviz(model_miss, out_file=None, feature_names=features.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
graph_miss = graphviz.Source(dot_data_miss)
graph_miss.render("decision_tree_Miss_Classification")
# graph_miss.view()


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
# plt.show()
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
# plt.show()
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
# plt.show()
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



X = df[['Age','GraduateOrNot','AnnualIncome','FamilyMembers','ChronicDiseases','FrequentFlyer','FrequentFlyer']]
y = df["TravelInsurance"]

Kmean = cluster.KMeans(n_clusters=10, n_init=10)
Kmean.fit(X)

SSE = Kmean.inertia_
df['Cluster'] = Kmean.labels_

centers = Kmean.cluster_centers_
cluster_sizes = np.bincount(Kmean.labels_)

print(f"SSE K-Means: {SSE}")

for i, center in enumerate(centers):
    cluster_size = cluster_sizes[i]
    print(f"Cluster {i+1} - Center: {center} - Size: {cluster_size}")

plt.scatter(X['Age'], X['AnnualIncome'], c=df['Cluster'], cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 2], marker='X', color='red', s=200, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.legend()
# plt.show()



X = df[['Age', 'GraduateOrNot', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases', 'FrequentFlyer', 'FrequentFlyer']]
y = df["TravelInsurance"]

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تست تعداد خوشه‌ها از 2 تا 10
for num_clusters in range(2, 11):
    # پیاده‌سازی الگوریتم DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    # افزودن برچسب‌های خوشه به دیتافریم
    df['Cluster'] = labels
    
    # مقدار SSE (Sum of Squared Errors)
    sse = 0
    for cluster_label in set(labels):
        cluster_points = df[df['Cluster'] == cluster_label].drop('Cluster', axis=1)
        cluster_center = cluster_points.mean()
        sse += ((cluster_points - cluster_center) ** 2).sum().sum()
    
    # نمایش شکل شماتیک
    plt.scatter(X['Age'], X['AnnualIncome'], c=df['Cluster'], cmap='viridis')
    plt.title(f'DBSCAN Clustering - {num_clusters} Clusters')
    plt.xlabel('Age')
    plt.ylabel('Annual Income')
    # plt.show()
    
    # نمایش مقادیر
    print(f"\n number of clusters: {num_clusters}")
    print(f"SSE: {sse}")
    print(" number of member in cluster:")
    print(df['Cluster'].value_counts())
    print("---------------------------")



def calculate_and_display_metrics(y_true, y_pred, algorithm_name):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics_dict = {
        'Algorithm': algorithm_name,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1
    }
    return metrics_dict

# Function to plot ROC curve
def plot_roc_curve(y_true, y_probs, algorithm_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{algorithm_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {algorithm_name}')
    plt.legend(loc="lower right")
    plt.show()

# Decision Tree with Gini
model_gini = DecisionTreeClassifier()
model_gini.fit(X_train, y_train)
y_pred_gini = model_gini.predict(X_test)
metrics_gini = calculate_and_display_metrics(y_test, y_pred_gini, 'Decision Tree (Gini)')
print(metrics_gini)
y_probs_gini = model_gini.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_gini, 'Decision Tree (Gini)')

# Decision Tree with Entropy
model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)
metrics_entropy = calculate_and_display_metrics(y_test, y_pred_entropy, 'Decision Tree (Entropy)')
print(metrics_entropy)
y_probs_entropy = model_entropy.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_entropy, 'Decision Tree (Entropy)')

# Decision Tree with Misclassification
model_miss = DecisionTreeClassifier(criterion='gini')
model_miss.fit(X_train, y_train)
y_pred_miss = model_miss.predict(X_test)
metrics_miss = calculate_and_display_metrics(y_test, y_pred_miss, 'Decision Tree (Miss Classification)')
print(metrics_miss)
y_probs_miss = model_miss.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_probs_miss, 'Decision Tree (Miss Classification)')

# SVM with Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
metrics_linear = calculate_and_display_metrics(y_test, y_pred_linear, 'SVM (Linear Kernel)')
print(metrics_linear)
y_probs_linear = svm_linear.decision_function(X_test)
plot_roc_curve(y_test, y_probs_linear, 'SVM (Linear Kernel)')

# SVM with Polynomial Kernel
svm_poly = SVC(kernel='poly')
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
metrics_poly = calculate_and_display_metrics(y_test, y_pred_poly, 'SVM (Polynomial Kernel)')
print(metrics_poly)
y_probs_poly = svm_poly.decision_function(X_test)
plot_roc_curve(y_test, y_probs_poly, 'SVM (Polynomial Kernel)')

# SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
metrics_rbf = calculate_and_display_metrics(y_test, y_pred_rbf, 'SVM (RBF Kernel)')
print(metrics_rbf)
y_probs_rbf = svm_rbf.decision_function(X_test)
plot_roc_curve(y_test, y_probs_rbf, 'SVM (RBF Kernel)')