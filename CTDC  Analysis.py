# -*- coding: utf-8 -*-
"""
Created on Tue May 28 04:50:01 2019

@author: tfost
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import graphviz

ctdc = pd.read_csv('CTDC Global Dataset 3 Sept 2018.csv', low_memory=False)
ctdc.shape

ctdc.gender.sample(5)
ctdc.groupby('gender').count()
ctdc.gender.value_counts()

plt.figure()
ctdc.gender.value_counts().plot(kind='bar')
plt.title('Gender of Trafficking Victims')
plt.xlabel('Gender')
plt.show()

ctdc.ageBroad.sample(10)
ctdc.ageBroad.value_counts()

plt.figure()
ctdc.ageBroad.value_counts().plot(kind='bar')
plt.title('Age of Trafficking Victims')
plt.xlabel('Age')
plt.show()

ctdc.loc[ctdc.ageBroad == '0--8', 'ageBroad'] = '00--08'
ctdc.loc[ctdc.ageBroad == '9--17', 'ageBroad'] = '09--17'
plt.figure()
ctdc.ageBroad.value_counts().sort_index().plot(kind='bar')
plt.title('Age of Trafficking Victims')
plt.xlabel('Age')
plt.show()

ctdc.citizenship.sample(10)
ctdc.citizenship.value_counts().sort_values(ascending=False)

# Clean up the data for analysis
ctdc2 = ctdc.copy()
ctdc2 = ctdc2.loc[ctdc.yearOfRegistration > 2016]
ctdc2 = ctdc2.drop(['meansOfControlConcatenated',
                    'isForcedLabour',
                    'isSexualExploit',
                    'isOtherExploit',
                    'isSexAndLabour',
                    'isForcedMarriage',
                    'isForcedMilitary',
                    'isOrganRemoval',
                    'typeOfLabourAgriculture',
                    'typeOfLabourAquafarming',
                    'typeOfLabourBegging',
                    'typeOfLabourConstruction',
                    'typeOfLabourDomesticWork',
                    'typeOfLabourHospitality',
                    'typeOfLabourIllicitActivities',
                    'typeOfLabourManufacturing',
                    'typeOfLabourMiningOrDrilling',
                    'typeOfLabourPeddling',
                    'typeOfLabourTransportation',
                    'typeOfLabourOther',
                    'typeOfLabourNotSpecified',
                    'typeOfLabourConcatenated',
                    'typeOfSexProstitution',
                    'typeOfSexPornography',
                    'typeOfSexRemoteInteractiveServices',
                    'typeOfSexPrivateSexualServices',
                    'typeOfSexConcatenated'], axis=1)

# Make a decision tree
for column in ctdc2.columns:
    if ctdc2[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        ctdc2[column] = le.fit_transform(ctdc2[column])

ctdc_target = ctdc2['typeOfExploitConcatenated']
target_labels = ['Forced Labour', 'Forced Marriage', 'Forced Military',
                 'Organ Removal', 'Other Exploit', 'Sex and Labour',
                 'Sexual Exploit']
ctdc3 = ctdc2.copy()
ctdc3 = ctdc3.drop(['typeOfExploitConcatenated'], axis=1)
X = ctdc3
y = ctdc_target
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
decision_tree = decision_tree.fit(X, y)

tree_data = tree.export_graphviz(decision_tree, out_file=None,
                                 feature_names=ctdc3.columns,
                                 class_names=target_labels,
                                 filled=True, rounded=True,
                                 special_characters=True)
graph = graphviz.Source(tree_data)
graph.render("CTDC Decision Tree", view=True)
