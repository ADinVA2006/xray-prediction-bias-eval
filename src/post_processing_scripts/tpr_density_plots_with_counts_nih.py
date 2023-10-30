import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# script to generate plots - could benifit from some additional automation
# example here is for NIH
class TPR_density_comparison_with_counts:


    def __init__(self):
        self.diagnosis = np.asarray(['Mass','Infiltration','Nodule','Pneumonia','Cardiomegaly','Emphysema','Pleural_Thickening','Effusion','Atelectasis','Consolidation','Pneumothorax','Fibrosis','Hernia','Edema'])

    
    def get_counts(self, joined_df, col_for_count):
                
        m_label_counts_test = []
        f_label_counts_test = []

        for val in self.diagnosis:
            m_label_counts_test.append(len(joined_df[(joined_df[col_for_count] == 'M') & (joined_df[val] == 1)]))
            f_label_counts_test.append(len(joined_df[(joined_df[col_for_count] == 'F') & (joined_df[val] == 1)]))
        
        f_min = min(number for number in m_label_counts_test if number > 0)
        m_min = min(number for number in f_label_counts_test if number > 0)

        overall_min = min([f_min, m_min])

        return np.asarray(m_label_counts_test), np.asarray(f_label_counts_test), overall_min

    
    def tpr_disparity_density(self, m_counts, f_counts, min_scale):

        # need df with tpr disparity and calculated totals for training and validation (two plots)
        # values come from tpr_determination.py
        
        f_tpr_indiv = np.asarray([-0.008614864865,-0.01751629035,-0.02313088951,-0.02407407407,0.03440860215,-0.0441426146,0.04566699124,0.05517488398,-0.05892083149,0.06086956522,0.08074534161,-0.1264726265,0.1363636364,0.1826625387])
        m_tpr_indiv = np.asarray([0.008614864865,0.01751629035,0.02313088951,0.02407407407,-0.03440860215,0.0441426146,-0.04566699124,-0.05517488398,0.05892083149,-0.06086956522,-0.08074534161,0.1264726265,-0.1363636364,-0.1826625387])

        f_tpr_merged = np.asarray([0.01841216216,-0.08945490656,-0.0002718235477,0.112962963,0.007139784946,-0.06281833616,0.0700097371,0.05401470084,-0.04769570986,0.1043478261,0.05279503106,-0.08905058905,0.1727272727,0.1950464396])
        m_tpr_merged = np.asarray([-0.01841216216,0.08945490656,0.0002718235477,-0.112962963,-0.007139784946,0.06281833616,-0.0700097371,-0.05401470084,0.04769570986,-0.1043478261,-0.05279503106,0.08905058905,-0.1727272727,-0.195046439])        

        sns.set_style("darkgrid")
        plt.scatter(x=self.diagnosis, y=m_tpr_indiv, s=m_counts/2, label = 'M - indiv', color='blue')
        plt.scatter(x=self.diagnosis, y=f_tpr_indiv, s=f_counts/2, label = 'F - indiv', marker='^', color='blue')
        plt.scatter(x=self.diagnosis, y=m_tpr_merged, s=m_counts/2, label = 'M - merged', color='orange')
        plt.scatter(x=self.diagnosis, y=f_tpr_merged, s=f_counts/2, label = 'F - merged', marker='^',color='orange')

        plt.title("NIH TPR Disparity")
        plt.xlabel("Detection Labels")
        plt.ylabel("TPR Disparity")
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=30, ha='right')
        plt.tight_layout()
        plt.legend()

        plt.savefig('plots/nih_tpr_disparity_val_density_with_totals_pad.png')
        

# probably also want numerical differences between genders for easier comparison
# then need to create and/or modify eval script to get tabular results that can be joined with reference data

if __name__ == '__main__':
    plot_comp = TPR_density_comparison_with_counts()

    test_split_ref = "<path to test file splits>/file_splits/test_split_nih.csv"
    true_preds = "<path to predictions>/post_analysis_03/all_preds/nih_merged_True_pretrained.csv"

    gender_column = 'Patient Gender'
    image_index = 'Image Index'

    details = pd.read_csv(test_split_ref)
    true_preds_df = pd.read_csv(true_preds)

    bias_eval_columns = details[[image_index, gender_column]]
    true_preds_joined = true_preds_df.join(bias_eval_columns.set_index(image_index), on='path', validate='1:1')

    m_counts, f_counts, min_scale = plot_comp.get_counts(true_preds_joined, col_for_count=gender_column)
    

    plot_comp.tpr_disparity_density(m_counts, f_counts, min_scale)
