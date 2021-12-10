import numpy as np
import math
from sklearn.metrics import confusion_matrix
import pandas as pd

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, n_tm_model,  num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        
        self.model = n_tm_model
        
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        

         
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        value_sd = np.std(value_list)
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
            #thre=thre+np.choice()
        return thre   
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        
        tp=np.sum(self.t_tr_corr) 
        fn=len(self.t_tr_corr)-np.sum(self.t_tr_corr)
        fp=np.sum(self.t_te_corr) 
        tn= len(self.t_te_corr)-np.sum(self.t_te_corr)
        
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        
        acc=(tp+tn)/(tp+tn+fn+fp)

        print("TP : ", tp )
        print("TN : ", tn)
        print("FN : ", fn )
        print("FP : ", fp)
        
        print("precision : ",precision)
        print("recall : ",recall)
        print("acc : ", acc)
        
        return precision, acc, recall

    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        
        print(v_name)
        
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        true_mem, true_nonmem, flase_mem, false_nonmem = 0, 0, 0, 0
        
        fin_mem=[]
        fin_nmem=[]
        
        idx_tp_mem=[]
        idx_fn_nmem=[]
        idx_fp_mem=[]
        idx_tn_nmem=[]
        
        mem_pred_membership, nmem_pred_membership =[], []

        pd_idx_tr_values=pd.concat([pd.DataFrame(self.t_tr_labels),pd.DataFrame(t_tr_values)],axis=1)
        pd_idx_tr_values.columns = ['label','loss']

        pd_idx_te_values=pd.concat([pd.DataFrame(self.t_te_labels),pd.DataFrame(t_te_values)],axis=1)
        pd_idx_te_values.columns = ['label','loss']


        correct_ratio=[]
        
      
        per_class_threshold = {}

     
        for num in range(self.num_classes):
            
            #print("Class : ", num)
            
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
                        
            per_class_threshold[num] = thre
                      
            true_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            true_nonmem += np.sum(t_te_values[self.t_te_labels==num]<thre)
            
            flase_mem += np.sum(t_tr_values[self.t_tr_labels==num]<thre)
            false_nonmem += np.sum(t_te_values[self.t_te_labels==num]>=thre)
            
            
            #print("Class : ",num," ",np.sum(t_tr_values[self.t_tr_labels==num]>=thre), "/", len(t_tr_values[self.t_tr_labels==num]))
            
                  
            
            #######################################################################
            idxs_label = pd_idx_tr_values.loc[(pd_idx_tr_values['label']==num)]
            
            item_index_mem = idxs_label.loc[(idxs_label['loss'] >= thre)]
            member_tp_index = item_index_mem.index
            idx_tp_mem.append(list(member_tp_index))
            
            item_index_fn_nmem = idxs_label.loc[(idxs_label['loss'] < thre)]
            nonmember_fn_index = item_index_fn_nmem.index
            idx_fn_nmem.append(list(nonmember_fn_index))

            #######################################################################
            idxs_label = pd_idx_te_values.loc[(pd_idx_te_values['label']==num)]
            
            item_index_fp_mem = idxs_label.loc[(idxs_label['loss'] >= thre)]
            fp_member_index = item_index_fp_mem.index
            idx_fp_mem.append(list(fp_member_index))
            
            item_index_tn_nmem = idxs_label.loc[(idxs_label['loss'] < thre)]
            tn_nmember_index = item_index_tn_nmem.index
            idx_tn_nmem.append(list(tn_nmember_index))
 

 
 
            mem_pred_membership.append( np.where( t_tr_values[self.t_tr_labels==num] >= thre, 1,0))
            nmem_pred_membership.append( np.where( t_te_values[self.t_te_labels==num] >= thre, 1,0))
          
        
        
        idx_tp_mem = [item for sublist in idx_tp_mem for item in sublist]
        idx_fn_nmem = [item for sublist in idx_fn_nmem for item in sublist]
        idx_fp_mem = [item for sublist in idx_fp_mem for item in sublist]
        idx_tn_nmem = [item for sublist in idx_tn_nmem for item in sublist]

                   
       
        mem_flat_list = [item for sublist in mem_pred_membership for item in sublist]
        nmem_flat_list = [item for sublist in nmem_pred_membership for item in sublist]
        
        pred_membership= np.concatenate([mem_flat_list,nmem_flat_list])


        true_memm=np.ones(len(mem_flat_list), dtype=np.bool)
        true_nmem=np.zeros(len(nmem_flat_list), dtype=np.bool)
        true_membership= np.concatenate([true_memm,true_nmem])

        
        tn, fp, fn, tp = confusion_matrix(true_membership, pred_membership).ravel()

        
        print("TP : ", tp )
        print("TN : ", tn)
        print("FN : ", fn )
        print("FP : ", fp)

        #print("Member identification : ", t_tr_mem/(len(self.t_tr_labels)))
        #print("Nonmember identification : ", t_te_non_mem/(len(self.t_te_labels)))
        
        
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        
        acc=(tp+tn)/(tp+tn+fn+fp)

        print("precision : ",precision)
        print("recall : ",recall)
        print("acc : ", acc)

        mem_inf_acc = 0.5*(true_mem/(len(self.t_tr_labels)+0.0) + true_nonmem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        
        if v_name == 'modified entropy':
            return idx_tp_mem, idx_fn_nmem, idx_fp_mem, idx_tn_nmem, correct_ratio, acc, precision, mem_inf_acc, recall, per_class_threshold, true_membership, pred_membership
        else:
            return idx_tp_mem, idx_fn_nmem, idx_fp_mem, idx_tn_nmem, correct_ratio, acc, precision, mem_inf_acc, recall


    
    def _mem_inf_benchmarks(self,all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            cc_precision, cc_acc, cc_recall = self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            conf_tp, conf_fn, conf_fp, conf_tn, conf_correct_ratio, conf_acc, conf_precision, conf_mem_inf_acc, conf_recall = self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            entr_tp, entr_fn, entr_fp, entr_tn, entr_correct_ratio, entr_acc, entr_precision, entr_mem_inf_acc, entr_recall = self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            mentr_tp, mentr_fn, mentr_fp, mentr_tn, mentr_correct_ratio, mentr_acc, mentr_precision, mentr_mem_inf_acc, mentr_recall,  mentr_per_class_threshold, true_membership, pred_membership = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return conf_tp, conf_fn, conf_fp, conf_tn, conf_correct_ratio, conf_acc, conf_precision, conf_mem_inf_acc, conf_recall, entr_tp, entr_fn, entr_fp, entr_tn, entr_correct_ratio, entr_acc, entr_precision, entr_mem_inf_acc, entr_recall, mentr_tp, mentr_fn, mentr_fp, mentr_tn, mentr_correct_ratio, mentr_acc, mentr_precision, mentr_mem_inf_acc, mentr_recall, mentr_per_class_threshold, true_membership, pred_membership, cc_precision, cc_acc, cc_recall
    
    
    
    #idx_tp_mem, idx_fn_nmem,idx_fp_mem,idx_tn_nmem, correct_ratio, acc, precision, mem_inf_acc, recall
