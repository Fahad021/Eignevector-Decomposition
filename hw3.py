
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Task-1: Load the crime dataset and store it as a matrix 

#data was first downloaded into my machine and saves as "data.txt"
#attribute was properly labeled from repository data description

crimedata = pd.read_csv('data.txt',sep=',',names=['state','county', 'community','communityname','fold','population','householdsize','racepctblack','racePctWhite','racePctAsian','racePctHisp','agePct12t21','agePct12t29','agePct16t24','agePct65up','numbUrban','pctUrban','medIncome','pctWWage','pctWFarmSelf','pctWInvInc','pctWSocSec','pctWPubAsst','pctWRetire','medFamInc','perCapInc','whitePerCap','blackPerCap','indianPerCap','AsianPerCap','OtherPerCap','HispPerCap','NumUnderPov','PctPopUnderPov','PctLess9thGrade','PctNotHSGrad','PctBSorMore','PctUnemployed','PctEmploy','PctEmplManu','PctEmplProfServ','PctOccupManu','PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr','FemalePctDiv','TotalPctDiv','PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par','PctTeen2Par','PctWorkMomYoungKids','PctWorkMom','NumIlleg','PctIlleg','NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10','PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10','PctSpeakEnglOnly','PctNotSpeakEnglWell','PctLargHouseFam','PctLargHouseOccup','PersPerOccupHous','PerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup','PctPersDenseHous','PctHousLess3BR','MedNumBR','HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt','PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal','OwnOccHiQuart','RentLowQ','RentMedian','RentHighQ','MedRent','MedRentPctHousInc','MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','NumStreet','PctForeignBorn','PctBornSameState','PctSameHouse85','PctSameCity85','PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop','PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop','ViolentCrimesPerPop'], encoding='latin-1',engine='python',na_values=['?'])

#datadrame was stored to excel file
writer = pd.ExcelWriter('output_2.xlsx')
crimedata.to_excel(writer,'Sheet1')
writer.save()

#feature matrix extraction and missing value replacement with mean
df=crimedata.drop(['state','county', 'community','communityname','fold','ViolentCrimesPerPop'], axis=1)
df=df.fillna(df.mean())
print(df)

#Task-2: Compute the eigenvectors and eigenvalues 
df=df.values
m = np.asmatrix(df)

print(m)
print(m.shape)

#eignevalue only defined for square matrix.Data is not square matrix
#values, vectors = np.linalg.eig(m)
#print(values)
#print(vectors)

#Singular value decomposition using numpy function
u, s, vh = np.linalg.svd(m, full_matrices=True)
#print(u.shapre),print(s.shape),print(vh.shape)
#SVD is usually described for the factorization of a 2D matrix A. 
#The higher-dimensional case will be discussed below. 
#In the 2D case, SVD is written as A = U S V^H, where A = a, U= u, S= \mathtt{np.diag}(s) and V^H = vh.
#The 1D array s contains the singular values of a and u and vh are unitary. 
#The rows of vh are the eigenvectors of A^H A and the columns of u are the eigenvectors of A A^H. 
#In both cases the corresponding (possibly non-zero) eigenvalues are given by s**2.


#task-3

plt.plot(s)

