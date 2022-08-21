import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler


# print(sys.argv[1])

def load_network(file_path):
    ppi = pd.read_table(filepath_or_buffer=file_path, header=None,
                        index_col=None, names=['source', 'target'], sep='\t')
    ppi_nodes = pd.concat([ppi['source'], ppi['target']], ignore_index=True)
    ppi_nodes = pd.DataFrame(ppi_nodes, columns=['nodes']).drop_duplicates()
    ppi_nodes.reset_index(drop=True, inplace=True)
    return ppi,ppi_nodes

def get_row_intersection_of_dataframe(df1,df2,df3):
    r=pd.DataFrame(df1.index.values.tolist(),columns=['rna'])
    s=pd.DataFrame(df2.index.values.tolist(),columns=['snp'])
    if df3 is not None:
        p=pd.DataFrame(df3.iloc[:,0].values.tolist(),columns=['ppi'])
        rp=pd.merge(left=r,right=p,left_on='rna',right_on='ppi',how='inner')
        rps=pd.merge(left=rp,right=s,left_on='rna',right_on='snp',how='inner')
    else:
        rps = pd.merge(left=r, right=s, left_on='rna', right_on='snp', how='inner')
    g_lst=rps['rna'].values.tolist()
    return g_lst

def get_col_intersection_of_dataframe(df1,df2):
    pls1=df1.columns.values.tolist()
    pls2=df2.columns.values.tolist()

    p_lst=[]
    for p in pls1:
        if p in pls2:
           p_lst.append(p)
    return p_lst

def filter_ppi_with_intersect_nodes(nodes_lst,ppi_df):
    g_lst_df=pd.DataFrame(nodes_lst,columns=['g1'])
    m1=pd.merge(left=ppi_df,right=g_lst_df,left_on='source',right_on='g1',how='left')
    m1.dropna(how='any',inplace=True)
    m1.drop(['g1'],axis=1,inplace=True)

    m2=pd.merge(left=m1,right=g_lst_df,left_on='target',right_on='g1',how='left')
    m2.dropna(how='any',inplace=True)
    m2.drop(['g1'],axis=1,inplace=True)
    return m2

def cal_outlying_gene_lst(df):
    expr_df_T = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    test=StandardScaler().fit_transform(expr_df_T)
    expr_df = pd.DataFrame(test.T, index=df.index, columns=df.columns)
    otly_g=[]
    expr_df_abs=expr_df.abs()
    for g,row in expr_df_abs.iterrows():
        if (row>2).any():
            otly_g.append(g)
    return otly_g

def prepare_intersection_data(cancerType):
    ppi,ppi_nodes=load_network('./data/ppi_index.txt')
    rna_file='./data/%s/orig_data/exp.txt' % cancerType
    snp_file='./data/%s/orig_data/mut.txt' % cancerType
    rna_df=pd.read_table(filepath_or_buffer=rna_file,header=0,index_col=0,sep='\t')
    snp_df=pd.read_table(filepath_or_buffer=snp_file,header=0,index_col=0,sep='\t')

    g_lst=get_row_intersection_of_dataframe(df1=rna_df,df2=snp_df,df3=ppi_nodes)

    r_g_inter=rna_df.loc[g_lst,:]
    s_g_inter=snp_df.loc[g_lst,:]

    p_lst=get_col_intersection_of_dataframe(r_g_inter,s_g_inter)

    rna_inter_df=r_g_inter.loc[:,p_lst]
    snp_inter_df=s_g_inter.loc[:,p_lst]

    ppi=filter_ppi_with_intersect_nodes(g_lst,ppi)
    ppi.to_csv(path_or_buf='./data/%s/Cancer_List/PPI.txt' % cancerType, sep='\t', header=False,index=False)


def cal_outlying_gene_lst(df):
    expr_df_T = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    test=StandardScaler().fit_transform(expr_df_T)
    expr_df = pd.DataFrame(test.T, index=df.index, columns=df.columns)
    otly_g=[]
    expr_df_abs=expr_df.abs()
    for g,row in expr_df_abs.iterrows():
        if (row>2).any():
            otly_g.append(g)
    return otly_g

def create_mutation_and_driver_matrices(cancerType):
    rna_df = pd.read_table(filepath_or_buffer='./data/%s/orig_data/exp.txt' % cancerType, header=0, index_col=0, sep='\t')
    snp_df = pd.read_table(filepath_or_buffer='./data/%s/orig_data/mut.txt' % cancerType, header=0, index_col=0, sep='\t')

    samp_lst = rna_df.columns.values.tolist()
    dic_p = {}

    for id, row in snp_df.iterrows():
        if row.name not in dic_p.keys():
            dic_p[row.name] = list(np.abs(row.values))
        else:
            print('error in cnv_df, duplicate mutation.')

    mut_t1 = pd.DataFrame(dic_p, index=samp_lst)
    mut_P = pd.DataFrame(mut_t1.values.T, index=mut_t1.columns, columns=mut_t1.index)

    gold_drivers = pd.read_table(filepath_or_buffer='./data/%s/Cancer_List/driver_list.txt' % cancerType, header=None, index_col=None,names=['name'])
    # gold_drivers = pd.read_table(filepath_or_buffer='./data/NCG_known_711.txt', header=None, index_col=None, names=['name'])
    for m in dic_p.keys():
        if m not in list(gold_drivers['name'].values):
            dic_p[m] = [0 for i in np.arange(0, len(samp_lst))]

    t1 = pd.DataFrame(dic_p, index=samp_lst)
    P_orig = pd.DataFrame(t1.values.T, index=t1.columns, columns=t1.index)
    otly_g_ls = cal_outlying_gene_lst(rna_df)
    ppi, ppi_nodes = load_network('./data/%s/Cancer_List/PPI.txt' % cancerType)
    remv_mut_ls = []
    for g in otly_g_ls:
        if g not in ppi_nodes['nodes'].values.tolist():
            remv_mut_ls.append(g)
    t1 = np.sum(mut_P.values, axis=1)
    P = P_orig.iloc[t1.nonzero()[0], :]
    for g in remv_mut_ls.copy():
        if g not in P.index.values.tolist():
            remv_mut_ls.remove(g)

    P = P.drop(remv_mut_ls, axis=0)

    p = pd.DataFrame(P.index.values.tolist(), columns=['PPI'])
    s = pd.DataFrame(snp_df.index.values.tolist(), columns=['snp'])
    ps = pd.merge(left=p, right=s, left_on='PPI', right_on='snp', how='inner')
    gene = ps['PPI'].values.tolist()
    snp = snp_df.loc[gene, :]
    exp = rna_df.loc[gene, :]

    P.to_csv(path_or_buf='./data/%s/Cancer_List/mut_driver.txt' % cancerType, sep='\t', header=True, index=True)
    exp.to_csv(path_or_buf='./data/%s/Cancer_List/exp.txt' % cancerType, sep='\t', header=True, index=True)
    snp.to_csv(path_or_buf='./data/%s/Cancer_List/mut.txt' % cancerType, sep='\t', header=True, index=True)


# cancerType = "BRCA"
cancerType = sys.argv[1]
prepare_intersection_data(cancerType=cancerType)
create_mutation_and_driver_matrices(cancerType=cancerType)





