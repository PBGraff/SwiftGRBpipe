from MLutils import *
from sklearn import tree,preprocessing,grid_search,cross_validation,ensemble
import math
import numpy as np
import os

Nenc = 0
Ntrees = 100
cv=True

# read in prior training data
filename = "newdata/test_sample_prior12_combined.txt"
xall,yall = readdata(filename,Nenc)

# split data
xtrain,xtest,ytrain,ytest = cross_validation.train_test_split(xall,yall,test_size=0.25)

if cv:
	# setup grid search parameters
	nsplits = [2,4,8,16,32]
	nfeats = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	grid_pars = {'min_samples_split':nsplits, 'max_features':nfeats}

	# setup classifiers and train and evaluate on test
	basemod = ensemble.RandomForestClassifier(n_estimators=Ntrees)
	clf = grid_search.GridSearchCV(basemod,grid_pars,verbose=2,cv=5,n_jobs=4)
	clf.fit(xtrain,ytrain)
	print '\nCross-validation best params: ',clf.best_params_
	print 'Train score = ',clf.best_score_
	print 'Test score = ',clf.score(xtest,ytest),'\n'

	ofp = open('RF_gridsearch_scores.txt','w')
	ofp.write('max_features\tmin_samples_split\tmean_validation_score\n')
	for gs in clf.grid_scores_:
		mf = gs.parameters['max_features']
		mss = gs.parameters['min_samples_split']
		mvs = gs.mean_validation_score
		ofp.write(repr(mf)+'\t'+repr(mss)+'\t'+repr(mvs)+'\n')
	ofp.close()

# run ensemble of trees on best parameters
if cv:
	RF1 = ensemble.RandomForestClassifier(n_estimators=Ntrees,min_samples_split=clf.best_params_['min_samples_split'],max_features=clf.best_params_['max_features'])
	RF2 = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=clf.best_params_['min_samples_split'],max_features=clf.best_params_['max_features']),n_estimators=Ntrees)
else:
	RF1 = ensemble.RandomForestClassifier(n_estimators=Ntrees,min_samples_split=4,max_features=8)
	RF2 = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=4,max_features=8),n_estimators=Ntrees)
RF1.fit(xtrain,ytrain)
RF2.fit(xtrain,ytrain)

# get prediction over ensemble
#y_pred = RF.predict_proba(xtest)
print 'Random Forest:'
print '    Accuracy = ',RF1.score(xtest,ytest)
print '    Feature importances = ',RF1.feature_importances_
print 'AdaBoost Random Forest:'
print '    Accuracy = ',RF2.score(xtest,ytest)
print '    Feature importances = ',RF2.feature_importances_

# print predictions
'''
ofp=open('DecisionTreePredictions.txt','w')
for i in range(len(yp)):
	for j in range(15+Nenc):
		ofp.write(repr(xtest[i][j])+'\t')
	ofp.write(repr(ytest[i])+'\t'+repr(yp[i])+'\n')
ofp.close()
'''

# train and print example tree
if cv:
	RF3 = tree.DecisionTreeClassifier(min_samples_split=clf.best_params_['min_samples_split'],max_features=clf.best_params_['max_features'])
else:
	RF3 = tree.DecisionTreeClassifier(min_samples_split=4,max_features=8)
RF3.fit(xtrain,ytrain)
print '\nExample tree score = ',RF3.score(xtest,ytest),'\n'
tree_out = tree.export_graphviz(RF3,out_file="tree.dot",feature_names=getNames(15+Nenc))
tree_out.close()
os.system('dot -Tpng tree.dot -o tree.png')

# read in evaluation data and calculate scores
newdists = ['newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_noevo_ndet27147.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed11_noevo_ndet26997.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed12_noevo_ndet29413.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed50_noevo_ndet24387.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed60_noevo_ndet26478.txt']
for i in range(len(newdists)):
	xval,ytrue = readdata(newdists[i],Nenc)
	ypred = RF1.predict_proba(xval)
	score = RF1.score(xval,ytrue)
	print 'Random Forest Score on data set ',i+1,' = ',score
	PrintPredictions('results/Swift_testprior12_enc0_RF_data'+repr(i+1)+'_all_pred.txt',xval,ytrue,ypred,method='forest')
	ypred = RF2.predict_proba(xval)
	score = RF2.score(xval,ytrue)
	print 'AdaBoost Score on data set ',i+1,' = ',score
	PrintPredictions('results/Swift_testprior12_enc0_AB_data'+repr(i+1)+'_all_pred.txt',xval,ytrue,ypred,method='forest')
