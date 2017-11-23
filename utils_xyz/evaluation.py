# xyz Nov 2017

import numpy as np

class EvaluationMetrics():
    @staticmethod
    def get_class_accuracy(c_TP_FN_FP,total_num,class_ls=None):
        c_TP_FN_FP = c_TP_FN_FP.astype(np.float)
        TP = c_TP_FN_FP[0]
        FN = c_TP_FN_FP[1]
        FP = c_TP_FN_FP[2]
        total_num = total_num*1.0
        precision = np.nan_to_num(TP/(TP+FP))
        recall = np.nan_to_num(TP/(TP+FN))
        IOU = np.nan_to_num(TP/(TP+FN+FP))
        # weighted ave
        real_Pos = TP+FN
        normed_real_TP = real_Pos/np.sum(real_Pos)
        ave_4acc = np.zeros(shape=(4),dtype=np.float)
        ave_4acc[0] = np.sum( precision*normed_real_TP )
        ave_4acc[1] = np.sum( recall*normed_real_TP )
        ave_4acc[2] = np.sum( IOU*normed_real_TP )
        ave_4acc[3] = np.sum(TP)/total_num
        ave_4acc_name = ['ave_class_pre','ave_class_rec','ave_class_IOU','ave_point_accu']
        # gen str
        delim = '' # ','
        def getstr(array,mean=None,str_format='%0.3g'):
            if mean!=None:
                mean_str = '%9s'%(str_format%mean) + delim
            else:
                mean_str = '%9s'%('  ')
                if delim != '': mean_str = mean_str + ' '

            return mean_str + delim.join(['%9s'%(str_format%v) for v in array])
        ave_class_acc_str = 'point average:  %0.3f,  class ave pre/rec/IOU: %0.3f/ %0.3f/ %0.3f    N = %f M'% \
            ( ave_4acc[3],ave_4acc[0],  ave_4acc[1], ave_4acc[2], total_num/1000000.0)
        if class_ls != None:
            class_acc_str = ave_class_acc_str + '\n\t       average'+delim  + delim.join(['%9s'%c for c in class_ls])+'\n'
        else:
            class_acc_str = ''
        class_acc_str += 'class_pre:   '+getstr(precision,ave_4acc[0])+'\n'
        class_acc_str += 'class_rec:   '+getstr(recall,ave_4acc[1])+'\n'
        class_acc_str += 'class_IOU:   '+getstr(IOU,ave_4acc[2])+'\n'
        class_acc_str += 'number(K):   '+getstr(np.trunc(real_Pos/1000.0),str_format='%d')
        return class_acc_str,ave_class_acc_str

    @staticmethod
    def get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label):
        c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))
        for i,pred_i in enumerate(pred_val):
            l = cur_label[i]
            c_TP_FN_FP[0,l] += pred_i==l
            c_TP_FN_FP[1,l] += pred_i!=l
            c_TP_FN_FP[2,pred_i] += pred_i!=l
        return c_TP_FN_FP
