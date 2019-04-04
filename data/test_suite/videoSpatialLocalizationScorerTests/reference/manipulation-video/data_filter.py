import pandas as pd

def read_csv(fname,sep="|"):
    return pd.read_csv(fname,sep=sep,header=0,index_col=False,na_filter=False)

def write_csv(fname,df,sep="|"):
    df.to_csv(fname,index=False,sep=sep)

if __name__ == '__main__':
    idx = read_csv("../../indexes/MFC18_Dev2-manipulation-video-index.csv")
    ref_name = "MFC18_Dev2-manipulation-video-ref.csv"
    ref = read_csv(ref_name)
    pjj_name = "MFC18_Dev2-manipulation-video-ref-probejournaljoin.csv"
    pjj = read_csv(pjj_name)
    jm_name = "MFC18_Dev2-manipulation-video-ref-journalmask.csv"
    jm = read_csv(jm_name)

    ref_cols = ref.columns.values.tolist()
    ref = ref.merge(idx)[ref_cols].drop_duplicates()
    write_csv(ref_name,ref)

    pjj_cols = pjj.columns.values.tolist()
    pjj = pjj.merge(ref)[pjj_cols].drop_duplicates()
    write_csv(pjj_name,pjj)

    jm_cols = jm.columns.values.tolist()
    jm = jm.merge(pjj)[jm_cols].drop_duplicates()
    write_csv(jm_name,jm)
    
