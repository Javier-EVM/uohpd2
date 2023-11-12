import numpy as np
import pandas as pd


def accuracy(pred,ground_truth):
    correct_pred = np.sum(pred==ground_truth) #predicciones correctas TP + TN
    return correct_pred / len(pred) #len(pred) total de predicciones TP + TN + FP + FN
    
def recall(pred,ground_truth):
    pred = np.array(pred)
    positives = np.where(pred==1)[0] #hay TP (VP) y FP (FP) correctamente clasificados como 1 
    #e incorrectamente clasificados como 1 
    true_positives = np.sum(pred[positives] == ground_truth[positives]) #reviso los TP
    total_positives = np.sum(pred==1) #el total de positivos = TP + FP

    if total_positives != 0:
        return true_positives / total_positives
    else:
        return 0

    #return true_positives / total_positives #SI ESTO DA NA, ES PORQUE EL MODELO NO PREDIJO VALORES 1.

def precision(pred,ground_truth): #creo que esta mal
    pred = np.array(pred)
    total_true = np.where(ground_truth==1)[0] #FN + TP
    positives = np.where(pred==1)[0] #TP + FP
    true_positives = np.sum(pred[positives] == ground_truth[positives]) #reviso los TP (ground_truth es 1)
    return true_positives / len(total_true)#el total de positi