import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

st.set_page_config(page_title="UPI Fraud Detection System", page_icon="🛡️", layout="wide")

@st.cache_data
def generate_data():
    np.random.seed(42); n=8000; nf=int(n*0.055)
    def legit(n):
        return pd.DataFrame({'amount':np.random.lognormal(6.5,1.2,n).clip(10,50000).round(2),
            'hour':np.random.choice(range(8,23),n),'day_of_week':np.random.randint(0,7,n),
            'txn_freq_7d':np.random.poisson(8,n).clip(1,40),
            'avg_txn_amt':np.random.lognormal(6.3,0.8,n).clip(50,20000).round(2),
            'device_age':np.random.randint(30,1500,n),
            'new_beneficiary':np.random.choice([0,1],n,p=[0.85,0.15]),
            'loc_mismatch':np.random.choice([0,1],n,p=[0.95,0.05]),
            'failed_attempts':np.random.choice([0,1,2],n,p=[0.90,0.08,0.02]),
            'vpn':np.random.choice([0,1],n,p=[0.97,0.03]),'label':0})
    def fraud(n):
        return pd.DataFrame({'amount':np.random.lognormal(8.5,1.5,n).clip(5000,200000).round(2),
            'hour':np.random.randint(0,24,n),'day_of_week':np.random.randint(0,7,n),
            'txn_freq_7d':np.random.poisson(25,n).clip(5,60),
            'avg_txn_amt':np.random.lognormal(7.5,1.2,n).clip(100,50000).round(2),
            'device_age':np.random.randint(1,60,n),
            'new_beneficiary':np.random.choice([0,1],n,p=[0.30,0.70]),
            'loc_mismatch':np.random.choice([0,1],n,p=[0.40,0.60]),
            'failed_attempts':np.random.choice([0,1,2,3],n,p=[0.30,0.25,0.25,0.20]),
            'vpn':np.random.choice([0,1],n,p=[0.55,0.45]),'label':1})
    return pd.concat([legit(n-nf),fraud(nf)],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)

@st.cache_resource
def train(df):
    feats=['amount','hour','day_of_week','txn_freq_7d','avg_txn_amt','device_age','new_beneficiary','loc_mismatch','failed_attempts','vpn']
    X=df[feats]; y=df['label']
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    m=GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.08,random_state=42)
    m.fit(Xt,yt)
    return m,feats,roc_auc_score(yv,m.predict_proba(Xv)[:,1]),confusion_matrix(yv,m.predict(Xv)),Xv,yv,m.predict_proba(Xv)[:,1]

df=generate_data(); model,feats,auc,cm,Xv,yv,yp=train(df)

st.title("🛡️ UPI Fraud Detection System")
st.caption("Real-time transaction risk scoring | Prajwal Markal Puttaswamy")
st.divider()

tab1,tab2,tab3=st.tabs(["🔍 Score Transaction","📊 Model Performance","📈 Fraud Patterns"])

with tab1:
    st.subheader("Real-Time Risk Scorer")
    c1,c2,c3=st.columns(3)
    with c1:
        amt=st.number_input("Amount (₹)",10,200000,15000,500)
        hr=st.slider("Hour",0,23,14)
        dow=st.selectbox("Day",['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    with c2:
        freq=st.slider("Txns last 7d",1,60,8)
        avg_amt=st.number_input("Avg Txn (₹)",50,50000,800)
        dev_age=st.slider("Device Age (days)",1,1500,365)
    with c3:
        nb=st.checkbox("New Beneficiary?")
        lm=st.checkbox("Location Mismatch?")
        fa=st.selectbox("Failed Attempts",[0,1,2,3])
        vpn=st.checkbox("VPN Detected?")
        if st.button("🛡️ Score Transaction",use_container_width=True):
            Xi=pd.DataFrame([[amt,hr,['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(dow),
                freq,avg_amt,dev_age,int(nb),int(lm),fa,int(vpn)]],columns=feats)
            prob=model.predict_proba(Xi)[0][1]; risk=prob*100
            c1,c2=st.columns(2)
            c1.metric("Risk Score",f"{risk:.1f}%")
            c2.metric("Decision","🚨 BLOCK" if prob>0.5 else ("⚠️ REVIEW" if prob>0.25 else "✅ APPROVE"))
            fig=go.Figure(go.Indicator(mode="gauge+number",value=risk,
                gauge={'axis':{'range':[0,100]},'bar':{'color':'#ff4a4a' if prob>0.5 else ('#f0c060' if prob>0.25 else '#4af0a0')},
                       'steps':[{'range':[0,25],'color':'#1a3a2a'},{'range':[25,50],'color':'#3a3a1a'},{'range':[50,100],'color':'#3a1a1a'}]}))
            fig.update_layout(height=250,paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
            if prob>0.5: st.error("🚨 HIGH RISK — Block transaction & flag account for review")
            elif prob>0.25: st.warning("⚠️ MEDIUM RISK — Trigger step-up authentication")
            else: st.success("✅ LOW RISK — Approve with standard monitoring")

with tab2:
    tn,fp,fn,tp=cm.ravel(); prec=tp/(tp+fp); rec=tp/(tp+fn)
    k1,k2,k3,k4=st.columns(4)
    k1.metric("ROC-AUC",f"{auc:.4f}"); k2.metric("Precision",f"{prec:.2%}")
    k3.metric("Recall",f"{rec:.2%}"); k4.metric("F1",f"{2*prec*rec/(prec+rec):.4f}")
    c1,c2=st.columns(2)
    with c1:
        fig=px.imshow(pd.DataFrame(cm,index=['Legit','Fraud'],columns=['Pred Legit','Pred Fraud']),
            text_auto=True,color_continuous_scale=['#111118','#c8f04a'],title="Confusion Matrix")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        tdf=Xv.copy(); tdf['label']=yv.values; tdf['prob']=yp
        fig=px.histogram(tdf,x='prob',color=tdf['label'].map({0:'Legit',1:'Fraud'}),
            nbins=40,barmode='overlay',title="Risk Score Distribution",
            color_discrete_map={'Legit':'#4af0a0','Fraud':'#ff4a4a'})
        st.plotly_chart(fig,use_container_width=True)

with tab3:
    fr=df[df['label']==1]; lg=df[df['label']==0]
    c1,c2=st.columns(2)
    with c1:
        fh=fr['hour'].value_counts().sort_index(); lh=lg['hour'].value_counts().sort_index()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=lh.index,y=lh.values/len(lg)*100,name='Legit',line=dict(color='#4af0a0')))
        fig.add_trace(go.Scatter(x=fh.index,y=fh.values/len(fr)*100,name='Fraud',line=dict(color='#ff4a4a',dash='dash')))
        fig.update_layout(title="Fraud vs Legit by Hour",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0.1)')
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.box(df,x=df['label'].map({0:'Legit',1:'Fraud'}),y='amount',
            color=df['label'].map({0:'Legit',1:'Fraud'}),log_y=True,title="Amount Distribution",
            color_discrete_map={'Legit':'#4af0a0','Fraud':'#ff4a4a'})
        st.plotly_chart(fig,use_container_width=True)
    st.caption("Built by Prajwal Markal Puttaswamy | [Portfolio](https://prajwalmarkalputtaswamyportfolio.netlify.app/)")
