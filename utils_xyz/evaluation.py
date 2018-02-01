# xyz Nov 2017

import numpy as np
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR+'/matterport_metadata')
from get_mpcat40 import MatterportMeta

class EvaluationMetrics():
    @staticmethod
    def get_class_accuracy(c_TP_FN_FP,total_num,class_ls=None,IsIncludeAveClass=False):
        c_TP_FN_FP = c_TP_FN_FP.astype(np.float)
        TP = c_TP_FN_FP[0]
        FN = c_TP_FN_FP[1]
        FP = c_TP_FN_FP[2]
        total_num = total_num*1.0
        ave_whole_acc = np.sum(TP)/total_num

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
        ave_class_num_weighted['recall'] = np.sum( recall*normed_real_TP )  # is equal to ave_whole_acc
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
             ave_class_num_weighted['IOU'],total_num/1000000.0, ave_whole_acc)
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
        return ave_whole_acc, class_acc_str,ave_class_acc_str

    @staticmethod
    def get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label):
        assert pred_val.shape == cur_label.shape
        c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))
        for i in range(pred_val.shape[0]):
            for j in range(pred_val.shape[1]):
                p = pred_val[i,j]
                l = cur_label[i,j]
                c_TP_FN_FP[0,l] += p==l
                c_TP_FN_FP[1,l] += p!=l
                c_TP_FN_FP[2,p] += p!=l
        return c_TP_FN_FP

    @staticmethod
    def report_pred( file_name, pred_val, label_category, dataset_name):
        if dataset_name == 'matterport3d':
            label2class = MatterportMeta['label2class']
        pred_logit = np.argmax( pred_val,-1 )
        is_correct = label_category == pred_logit
        print('writing pred log:%s'%(file_name))
        with open(file_name,'w') as f:
            classes_str = ' '.join([label2class[l] for l in range(len(label2class))])+'\n'
            for i in range(pred_val.shape[0]):
                for j in range(pred_val.shape[1]):
                    if is_correct[i,j]:
                        str_ij = '  Y '
                    else:
                        str_ij = '! N '
                    str_ij += '%s'%( label2class[label_category[i,j]] ) + '\t'
                    score_order = np.argsort( pred_val[i,j] )
                    for k in range( min(7,len(score_order)) ):
                        label_k = score_order[len(score_order) - k -1]
                        str_ij += '%s:%0.1f'%(label2class[label_k],pred_val[i,j,label_k]) + '  '
                    str_ij += '\n'
                    f.write(str_ij)
                f.flush()
        print('finish pred log:%s'%(file_name))

