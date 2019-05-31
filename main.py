
import pandas as pd
import numpy as np
import cufflinks as cf
from ipywidgets import interact, interact_manual
import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn.metrics import roc_curve, auc


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


def go():
    """Do all the work"""

    from plotly import __version__
    from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
    init_notebook_mode(connected=True)
    cf.go_offline()
    df = pd.DataFrame()
    df["Doesn't have disease"]=[1,3,5,8,8,9,9,9,14,15]
    df['Have Disease']=[10,14,15,17,17,18,18,19,20,21]
    true_predictions = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]

    @interact
    def scatter_plot(GraphColour=list(cf.colors._scales_names.keys()),Cuttoff_Val= np.arange(1,22)):

    #Calculating TN, FN, TP, FP & predictions (0 or 1) at your given cutoff value 
        tn,fn,tp,fp=0,0,0,0
        pred=[]
        for i in df["Doesn\'t have disease"]:
            if i <= Cuttoff_Val:
                tn=tn+1
                pred.append(0)
            else:
                fn=fn+1
                pred.append(1)    
        for i in df['Have Disease']:
            if i > Cuttoff_Val:
                tp=tp+1
                pred.append(1)
            else:
                fp=fp+1
                pred.append(0)

    #Account for division by zero and then calculate Sensitivity & Specificity
        if tp <= 0:
            sensitivity = 0
        else:
            sensitivity = tp/(tp+fn)

        if tn <= 0:
            speciificity = 0
        else:
            specificity = tn/(fp+tn)

    #Disease prevalence
        DP = (tp+fn)/(tp+tn+fp+fn)
    #Accuracy
        Acc = (tp+tn)/(tp+tn+fp+fn)
    #positive predicted value
        PPV = tp/(tp+fp)
    #Negative predicted value
        NPV = tn/(fn+tn)
        #Distribution Plot
        df.iplot(kind='hist',xTitle='Test Score', yTitle='Frequency',histfunc='count', barmode='overlay',opacity=0.75,
                 x=df["Doesn't have disease"],y=df['Have Disease'], colorscale=GraphColour,vline= Cuttoff_Val,annotations=
                 [dict(x=Cuttoff_Val+3.4,y=7,xref='x',yref='y',text='Sensitivity: '+ str(round(sensitivity*100,2))+
                '% <br>' +'Specificity: '+ str(round(specificity*100,2))+'% <br>' +'Desease Prevalence: '+
                str(round(DP*100,2))+'% <br>' +'PPV: '+ str(round(PPV*100,2))+'% , NPV: '+str(round(NPV*100,2))+'% <br>'+
                'Accuracy: '+ str(round(Acc*100,2))+'%',textangle= 360,showarrow=False,arrowhead=7,ax=0,ay=-40,)])

        #ROC curve calculations, plot settings and plotting
        fig_size = plt.rcParams["figure.figsize"]  
        fig_size[0] = 15 
        fig_size[1] = 10 
        plt.rcParams["figure.figsize"] = fig_size  
        fpr, tpr, thresholds = roc_curve(pred,true_predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
