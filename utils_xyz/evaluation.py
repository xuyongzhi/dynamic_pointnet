# xyz Nov 2017

import numpy as np

class EvaluationMetrics():
    @staticmethod
    def get_class_accuracy(c_TP_FN_FP,total_num,class_ls=None,IsIncludeAveClass=False):
        c_TP_FN_FP = c_TP_FN_FP.astype(np.float)
        TP = c_TP_FN_FP[0]
        FN = c_TP_FN_FP[1]
        FP = c_TP_FN_FP[2]
        total_num = total_num*1.0
        ave_whole_points = np.sum(TP)/total_num

        precision = np.nan_to_num(TP/(TP+FP))
        recall = np.nan_to_num(TP/(TP+FN))
        IOU = np.nan_to_num(TP/(TP+FN+FP))
        # class ave accuracy
        ave_class = {}
        ave_class['pre'] = np.mean(precision)
        ave_class['recall'] = np.mean(recall)
        ave_class['IOU'] = np.mean(IOU)
        # class num weighted ave
        real_Pos = TP+FN
        normed_real_TP = real_Pos/np.sum(real_Pos)
        ave_class_num_weighted = {}
        ave_class_num_weighted['pre'] = np.sum( precision*normed_real_TP )
        ave_class_num_weighted['recall'] = np.sum( recall*normed_real_TP )  # is equal to ave_whole_points
        ave_class_num_weighted['IOU'] = np.sum( IOU*normed_real_TP )

        # gen str
        delim = '' # ','
        def getstr(array,mean=None,str_format='%0.3g'):
            if mean!=None:
                mean_str = '%9s'%(str_format%mean) + delim
            else:
                mean_str = '%9s'%('  ')
                if delim != '': mean_str = mean_str + ' '

            return mean_str + delim.join(['%9s'%(str_format%v) for v in array])
        ave_class_acc_str = 'weighted class pre/rec/IOU: %0.3f  %0.3f  %0.3f  N=%fM  whole points average:  %0.3f'% \
            ( ave_class_num_weighted['pre'], ave_class_num_weighted['recall'],
             ave_class_num_weighted['IOU'],total_num/1000000.0, ave_whole_points)
        if IsIncludeAveClass:
            ave_class_acc_str += '\nclass ave pre/rec/IOU : %0.3f/ %0.3f/ %0.3f' %(
                        ave_class['pre'],ave_class['recall'],ave_class['IOU'])
        if class_ls != None:
            class_acc_str = ave_class_acc_str + '\n\t       average'+delim  + delim.join(['%9s'%c for c in class_ls])+'\n'
        else:
            class_acc_str = ''
        class_acc_str += 'class_pre:   '+getstr(precision,ave_class_num_weighted['pre'])+'\n'
        class_acc_str += 'class_rec:   '+getstr(recall,ave_class_num_weighted['recall'])+'\n'
        class_acc_str += 'class_IOU:   '+getstr(IOU,ave_class_num_weighted['IOU'])+'\n'
        class_acc_str += 'number(K):   '+getstr(np.trunc(real_Pos/1000.0),str_format='%d')
        return class_acc_str,ave_class_acc_str

    @staticmethod
    def get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label):
        c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))
        for i in range(pred_val.shape[0]):
            for j in range(pred_val.shape[1]):
                p = pred_val[i,j]
                l = cur_label[i,j]
                c_TP_FN_FP[0,l] += p==l
                c_TP_FN_FP[1,l] += p!=l
                c_TP_FN_FP[2,p] += p!=l
        return c_TP_FN_FP
