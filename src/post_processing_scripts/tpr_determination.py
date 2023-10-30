
import json

import pandas as pd
from sklearn.metrics import confusion_matrix


categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# age_decile = ['0-20', '20-40', '40-60', '60-80', '80-']
age_decile = ['40-60', '60-80', '20-40', '80-', '0-20']
gender = ['M', 'F']

def main():
    print("TPRs calculated")
    exit()

# example for PAD data set
if __name__ == "__main__":
    path_for_tprs = '/post_analysis_03/model_tprs/'
    dataset_descriptor = 'pad_merged_'
    bipreds_file = '/post_analysis_03/all_preds/pad_merged_bipred_pretrained.csv'
    true_preds = '/post_analysis_03/all_preds/pad_merged_True_pretrained.csv'
    test_split_ref = "/file_splits/test_split_pad_chest.csv" # may need to pull from TorchXRayVision
    
    # data source specific column names
    gender_column = 'PatientSex_DICOM'
    image_index = 'ImageID'

    details = pd.read_csv(test_split_ref)
    true_preds_df = pd.read_csv(true_preds)
    bipreds_df = pd.read_csv(bipreds_file)

    bias_eval_columns = details[[image_index, gender_column]]
    
    true_preds_joined = true_preds_df.join(bias_eval_columns.set_index(image_index), on='path', validate='1:1')
    bipreds_joined = bipreds_df.join(bias_eval_columns.set_index(image_index), on='path', validate='1:1')

    female_TPR = {}
    male_TPR = {}

    for diagnosis in categories:
        # df2=df.loc[df['Fee'] == 30000, 'Courses']
        y_true_m = true_preds_joined.loc[true_preds_joined[gender_column] == 'M', diagnosis].to_list()
        y_prediction_m = bipreds_joined.loc[bipreds_joined[gender_column] == 'M','bi_' + diagnosis].to_list()
        cnf_matrix_m = confusion_matrix(y_true_m, y_prediction_m) 

        TN_m, FP_m, FN_m, TP_m = cnf_matrix_m.ravel()

        TP_m = TP_m.astype(float)
        FN_m = FN_m.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR_m = TP_m/(TP_m+FN_m)

        male_TPR[diagnosis] = TPR_m

        # df2=df.loc[df['Fee'] == 30000, 'Courses']
        y_true_f = true_preds_joined.loc[true_preds_joined[gender_column] == 'F', diagnosis].to_list()
        y_prediction_f = bipreds_joined.loc[bipreds_joined[gender_column] == 'F','bi_' + diagnosis].to_list()
        cnf_matrix_f = confusion_matrix(y_true_f, y_prediction_f) 

        TN_f, FP_f, FN_f, TP_f = cnf_matrix_f.ravel()

        TP_f = TP_f.astype(float)
        FN_f = FN_f.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR_f= TP_f/(TP_f+FN_f)

        female_TPR[diagnosis] = TPR_f 

    with open(path_for_tprs + dataset_descriptor + "male_tprs.txt", 'w') as f: 
        for key, value in male_TPR.items(): 
            f.write('%s:%s\n' % (key, value))
    
    with open(path_for_tprs + dataset_descriptor + "female_tprs.txt", 'w') as f: 
        for key, value in female_TPR.items(): 
            f.write('%s:%s\n' % (key, value))

    main()
