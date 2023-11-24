import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

excelFile='G:/uni/DataMining/pythoncode/TravelInsurancePrediction.xlsx'
df=pd.read_excel(excelFile)

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

maxAge=df[columnName2].max()
minAge=df[columnName2].min()
medianAge=df[columnName2].median()
print("max Age:",maxAge)
print("min Age:",minAge)
print("median Age:", medianAge)

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
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('heatmap of TravelInsurancePrediction')
plt.show()


AgeCount=df[columnName2].value_counts().sort_index()
plt.bar(AgeCount.index,AgeCount.values)
plt.xlabel('Age')
plt.ylabel('count')
plt.title('Age Distribution')
plt.show()


employment_counts = df[['EmploymentType_Government Sector', 'EmploymentType_Private Sector/Self Employed']].sum()
plt.bar(employment_counts.index,employment_counts.values)
plt.xlabel('Employment Type')
plt.ylabel('count')
plt.title('Employment Type Distribution')
plt.show()

GraduateOrNotCount=df[columnName4].value_counts()
plt.bar(GraduateOrNotCount.index,GraduateOrNotCount.values)
plt.xlabel('GraduateOrNot')
plt.ylabel('count')
plt.title('GraduateOrNot Distribution')
plt.show()

df['LogAnnualIncome']=np.log10(df['AnnualIncome'])
plt.bar(df['LogAnnualIncome'].value_counts().index,df['LogAnnualIncome'].value_counts().values)
plt.xlabel('AnnualIncome')
plt.ylabel('count')
plt.title('AnnualIncome Distribution')
plt.show()

FamilyMembersCount=df[columnName6].value_counts().sort_index()
plt.bar(FamilyMembersCount.index,FamilyMembersCount.values)
plt.xlabel('FamilyMembers')
plt.ylabel('count')
plt.title('FamilyMembers Distribution')
plt.show()

ChronicDiseasesCount=df[columnName7].value_counts()
plt.bar(ChronicDiseasesCount.index,ChronicDiseasesCount.values)
plt.xlabel('ChronicDiseases')
plt.ylabel('count')
plt.title('ChronicDiseases Distribution')
plt.show()

FrequentFlyerCount=df[columnName8].value_counts()
plt.bar(FrequentFlyerCount.index,FrequentFlyerCount.values)
plt.xlabel('FrequentFlyer')
plt.ylabel('count')
plt.title('FrequentFlyer Distribution')
plt.show()

EverTravelledAbroadCount=df[columnName9].value_counts()
plt.bar(EverTravelledAbroadCount.index,EverTravelledAbroadCount.values)
plt.xlabel('EverTravelledAbroad')
plt.ylabel('count')
plt.title('EverTravelledAbroad Distribution')
plt.show()

TravelInsuranceCount=df[columnName10].value_counts()
plt.bar(TravelInsuranceCount.index,TravelInsuranceCount.values)
plt.xlabel('TravelInsurance')
plt.ylabel('count')
plt.title('TravelInsurance Distribution')
plt.show()