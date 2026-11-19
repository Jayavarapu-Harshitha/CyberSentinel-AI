import time
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,balanced_accuracy_score

def metrics(y, probabilities, threshold=.5):
    y=np.asarray(y); p=np.asarray(probabilities); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    result={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'pr_auc':average_precision_score(y,p) if len(np.unique(y))==2 else None,'roc_auc':roc_auc_score(y,p) if len(np.unique(y))==2 else None,'false_positive_rate':fp/(fp+tn) if fp+tn else None,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    return {k:float(v) if isinstance(v,(float,np.floating)) else v for k,v in result.items()}

def timed_probabilities(model,x,repeats=3):
    model.predict_proba(x.iloc[:min(len(x),10)]) # warm-up
    times=[]; probs=None
    for _ in range(repeats):
        start=time.perf_counter(); probs=model.predict_proba(x)[:,1]; times.append(time.perf_counter()-start)
    return probs,{'inference_seconds_median':float(np.median(times)),'inference_ms_per_row_median':float(np.median(times)*1000/len(x)),'inference_rows':len(x)}
