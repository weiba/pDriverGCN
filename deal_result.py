import pandas as pd
import numpy as np
import sys

cancerType = sys.argv[1]
# snp = pd.read_table(filepath_or_buffer='./data/%s/NCG_711/mut.txt' % cancerType, header=0, index_col=0, sep='\t')
# final_result = pd.read_table(filepath_or_buffer='./data/%s/NCG_711/result/final_result.txt' % cancerType, header=0, index_col=0, sep='\t')

snp = pd.read_table(filepath_or_buffer='./data/%s/Cancer_List/mut.txt' % cancerType, header=0, index_col=0, sep='\t')
final_result = pd.read_table(filepath_or_buffer='./data/%s/Cancer_List/result/final_result.txt' % cancerType, header=0, index_col=0, sep='\t')


patient_genes = pd.DataFrame()
for patient in final_result.columns.values:
    # print(patient)
    patient_score = pd.DataFrame(final_result.loc[:,patient])
    patient_mut = pd.DataFrame(snp.loc[:,patient])
    patient_mut_gene = np.where(patient_mut.values != 0)[0]
    patient_mut_score = patient_score.iloc[patient_mut_gene, :]
    patient_mut_sort = patient_mut_score.sort_values(by=patient, ascending=False)
    patient_sort_gene = patient_mut_sort.index.values
    if patient_sort_gene.shape[0]<2:
        patient_sort_gene1 = patient_sort_gene
    else:
        patient_sort_gene1 = patient_sort_gene[:int(patient_sort_gene.shape[0]/2)]
    patient_sort_gene2 = pd.DataFrame(patient_sort_gene1,columns=[patient])
    patient_genes = pd.concat([patient_genes,patient_sort_gene2], axis=1)

# patient_genes.T.to_csv(path_or_buf='./data/%s/NCG_711/result/sort_gene.txt' % cancerType,sep="\t")
patient_genes.T.to_csv(path_or_buf='./data/%s/Cancer_List/result/sort_gene.txt' % cancerType,sep="\t")

