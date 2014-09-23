from MLutils import *
from sklearn import svm,preprocessing,grid_search,cross_validation
import math
from numpy import std

Nenc = 0

# read in prior training data
filename = "newdata/test_sample_prior12_combined.txt"
xtrain,ytrain = readdata(filename,Nenc)

# split data
#xtrain,xtest,ytrain,ytest = cross_validation.train_test_split(xall,yall,test_size=0.5)

# scale data
scaler = preprocessing.StandardScaler()
xtrain = scaler.fit_transform(xtrain)
#xtest  = scaler.transform(xtest)

# setup hyperparameter search
lgmin = -4
lgmax = 4
lcmin = -4
lcmax = 4
ng = 10
nc = 10
cs=map(lambda x:10**(x*(lcmax-lcmin)/(nc-1.0)+lcmin),range(nc))
gs=map(lambda x:10**(x*(lgmax-lgmin)/(ng-1.0)+lgmin),range(ng))
grid_pars = {'C':cs, 'gamma':gs}
#print grid_pars

# train classifier
svr = svm.SVC(kernel='rbf')
clf = grid_search.GridSearchCV(svr,grid_pars,verbose=2,cv=5,n_jobs=4)
clf.fit(xtrain, ytrain)

# print scores
print "Grid scores:"
print "\n score>0.96:"
for v in clf.grid_scores_:
    c=v.parameters['C']
    g=v.parameters['gamma']
    if(v.mean_validation_score>0.95):
        print c,g,":",v.mean_validation_score,std(v.cv_validation_scores)

ofp = open('SVM_gridsearch_scores.txt','w')
ofp.write('C\tgamma\tmean_validation_score\n')
for gs in clf.grid_scores_:
	c = gs.parameters['C']
	g = gs.parameters['gamma']
	mvs = gs.mean_validation_score
	ofp.write(repr(c)+'\t'+repr(g)+'\t'+repr(mvs)+'\n')
ofp.close()

print
print "Best params:",clf.best_params_,"Best score=",clf.best_score_#,"Test score=",clf.score(xtest,ytest)
print 

# read in evaluation data and calculate scores
newdists = ['newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_noevo_ndet27147.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed11_noevo_ndet26997.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed12_noevo_ndet29413.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed50_noevo_ndet24387.txt',\
'newdata/summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_seed60_noevo_ndet26478.txt']
i=1
for filename2 in newdists:
	xval,ytrue = readdata(filename2,Nenc)
	xval = scaler.transform(xval)
	score = clf.best_estimator_.score(xval,ytrue)
	ypred = clf.best_estimator_.predict(xval)
	xval = scaler.inverse_transform(xval)
	PrintPredictions('results/Swift_testprior12_enc0_SVM_data'+repr(i)+'_all_pred.txt',xval,ytrue,ypred,method='svm')
	print 'Score on data set ',i,' = ',score
	i=i+1
