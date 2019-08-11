#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:34:31 2017

@author: jumingxing & xiaolei
"""

import numpy as np
from sklearn import ensemble,metrics

##################### INS metrics calculation
clf_final_v1.fit(train,target)

target_ins = clf_final_v1.predict_proba(train)
target_ins = np.array(target_ins)
rf_results = pd.DataFrame({'prediction':target_ins[:,1],"label":target})
ks_dis = calc_ks(rf_results, 10, prediction="prediction")
print(max(ks_dis))
ks_cont = calc_continus_ks(rf_results, prediction="prediction")
print(max(ks_cont))

preds = clf_final_v1.predict(train)
probas = clf_final_v1.predict_proba(train)
ACC = metrics.accuracy_score(target,preds)
precision, recall, thresholds = metrics.precision_recall_curve(target, probas[:, 1])
fpr_train, tpr_train, thresholds = metrics.roc_curve(target, probas[:, 1])
ROC = metrics.auc(fpr, tpr)
cm = metrics.confusion_matrix(target,preds)
print("ACCURACY:", ACC)
print("ROC:", ROC)
print("F1 Score:", metrics.f1_score(target,preds))
print( "TP:", cm[1,1], cm[1,1]/(cm.sum()+0.0))
print( "FP:", cm[0,1], cm[0,1]/(cm.sum()+0.0))
print( "Precision:", cm[1,1]/(cm[1,1]+cm[0,1]*1.1))
print( "Recall:", cm[1,1]/(cm[1,1]+cm[1,0]*1.1))

##################### OOS metrics calculation
target_oos = clf_final_v1.predict_proba(test)
target_oos = np.array(target_oos)
rf_results = pd.DataFrame({'prediction':target_oos[:,1],"label":real_target})
ks_dis = calc_ks(rf_results, 10, prediction="prediction")
print(max(ks_dis))
ks_cont = calc_continus_ks(rf_results, prediction="prediction")
print(max(ks_cont))

preds = clf_final_v1.predict(test)
probas = clf_final_v1.predict_proba(test)
ACC = metrics.accuracy_score(real_target,preds)
precision, recall, thresholds = metrics.precision_recall_curve(real_target, probas[:, 1])
fpr_test, tpr_test , thresholds = metrics.roc_curve(real_target, probas[:, 1])
ROC = metrics.auc(fpr, tpr)
cm = metrics.confusion_matrix(real_target,preds)
print("ACCURACY:", ACC)
print("ROC:", ROC)
print("F1 Score:", metrics.f1_score(real_target,preds))
print( "TP:", cm[1,1], cm[1,1]/(cm.sum()+0.0))
print( "FP:", cm[0,1], cm[0,1]/(cm.sum()+0.0))
print( "Precision:", cm[1,1]/(cm[1,1]+cm[0,1]*1.1))
print( "Recall:", cm[1,1]/(cm[1,1]+cm[1,0]*1.1))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_train, tpr_train, label='train')
plt.plot(fpr_test, tpr_test, label='test')


plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()