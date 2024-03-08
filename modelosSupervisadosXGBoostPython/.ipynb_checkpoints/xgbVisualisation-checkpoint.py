#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xgboost
import matplotlib.pyplot as plt
import shap


# In[ ]:


### SHAP visualisation
def make_shap_visualisation(finalXGB, xtrain):
    explainer = shap.TreeExplainer(finalXGB)
    shap_values = explainer.shap_values(xtrain)
    shap.summary_plot(shap_values, xtrain)


# In[ ]:


### XGB Native variable importance 

def plotVariableImportance(finalXGB, viz_type=None):
    
    if viz_type == None:
        xgboost.plot_importance(finalXGB)
        plt.title("xgboost.plot_importance(model)")
        plt.show()
        
    if viz_type == "cover":
        
        xgboost.plot_importance(finalXGB, importance_type="cover")
        plt.title('xgboost.plot_importance(model, importance_type="cover")')
        plt.show()
        
    if type == "gain":
        xgboost.plot_importance(finalXGB, importance_type="gain")
        plt.title('xgboost.plot_importance(model, importance_type="gain")')
        plt.show()


# In[1]:


# visualize training error metrics 

def visualise_metrics(finalXGB, mod):
    
    if mod == 'classification':
        results = finalXGB.evals_result_
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(range(len(results['validation_0']['auc'])), results['validation_0']['auc'], label='train auc')
        ax.plot(range(len(results['validation_1']['auc'])), results['validation_1']['auc'], label='test auc')
        ax.legend()
        plt.ylabel("auc")
        plt.title('AUC at Training')
        plt.show()
        
    if mod == 'regression':
    
        results = finalXGB.evals_result_
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(range(len(results['validation_0']['rmse'])), results['validation_0']['rmse'], label='train rmse')
        ax.plot(range(len(results['validation_1']['rmse'])), results['validation_1']['rmse'], label='test rmse')
        ax.legend()
        plt.ylabel("rmse")
        plt.title('RMSE at Training')
        plt.show()

