import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as Lr
from sklearn.metrics import mean_squared_error as MSE
import math
from scipy.stats import norm



WndSize=1
Clmn='Close'#['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],

# Data is from the Yahoo Finance
Dow=pd.read_csv('Dow.csv',index_col='Date')
Ftse=pd.read_csv('FTSE.csv',index_col='Date')
Nsdq=pd.read_csv('Nasdaq.csv',index_col='Date')
Nfty=pd.read_csv('Nifty50.csv',index_col='Date')
Nkki=pd.read_csv('Nikkei.csv',index_col='Date')
Dax=pd.read_csv('DAX.csv',index_col='Date')
Cac=pd.read_csv('CAC.csv',index_col='Date')
HngSng=pd.read_csv('HangSeng.csv',index_col='Date')
SngiCmpst=pd.read_csv('ShanghaiComposite.csv',index_col='Date')
#CrdOil=pd.read_csv('CrudeOil.csv',index_col='Date')
Ksp=pd.read_csv('Kospi.csv',index_col='Date')
Sp=pd.read_csv('SandP.csv',index_col='Date')

# Computing the moving average
MovAvg=pd.DataFrame()
MovAvg['Dow']=Dow[Clmn].rolling(window=WndSize).mean()
MovAvg['Nasdaq']=Nsdq[Clmn].rolling(window=WndSize).mean()
MovAvg['Nifty']=Nfty[Clmn].rolling(window=WndSize).mean()
MovAvg['Nikkei']=Nkki[Clmn].rolling(window=WndSize).mean()
MovAvg['Dax']=Dax[Clmn].rolling(window=WndSize).mean()
MovAvg['Cac']=Cac[Clmn].rolling(window=WndSize).mean()
MovAvg['HangSeng']=HngSng[Clmn].rolling(window=WndSize).mean()
MovAvg['Kospi']=Ksp[Clmn].rolling(window=WndSize).mean()
MovAvg['SanghiComposite']=SngiCmpst[Clmn].rolling(window=WndSize).mean()
#MovAvg['SP']=Sp[Clmn].rolling(window=WndSize).mean()
#MovAvg['CrdOil']=CrdOil[Clmn].rolling(window=WndSize).mean()
MovAvg=MovAvg.loc['2015-06-08':]
MovAvg=MovAvg.dropna()
Aprx='Nifty'
Frm=['Dow','Nifty','Nasdaq','Dax','Cac','Kospi','Nikkei','HangSeng','SanghiComposite']
X=MovAvg.loc[:'2020-06-02',Frm]
y=MovAvg.loc['2015-06-09':,Aprx]

"""
for i in range(len(X)):
    for j in range(len(Frm)):
        if math.isnan(X.iloc[i,j])==True:
            X.iloc[i,j]=(X.iloc[i-1,j]+X.iloc[i+1,j])/2.0
    if math.isnan(y.iloc[i])==True:
        y.iloc[i]=(y.iloc[i-1]+y.iloc[i+1])/2.0
"""
model=Lr().fit(X,y)
Xtest=np.array([[27110.98,10142.15,9814.08,12847.68,5197.79,2181.87,22863.73,24770.41,2930.80]])
Ypred=model.predict(Xtest)

print(Ypred)
"""
# Visualizarion of Error
Err=abs(y-Ypred)
mu=np.mean(Err)
std=np.std(Err)

ax=plt.axes()
gx = np.linspace(-1000,1100, 500)
gy = norm.pdf(gx, mu, std)
ax.plot(gx, gy, 'k', linewidth=2)


ax.hist(Err,bins=100,histtype='barstacked')#,density=True)
#ax.plot(gx,gy,'r')
ax.set_title('Histogram of prediction error at each trading day')
ax.set_xlabel('Error')
ax.set_ylabel('Number of days')
ax.text( 400,80, r'Mean:'+'{:.2f}'.format(mu), fontsize=10)
ax.text( 400,70, r'Standard deviation:'+'{:.2f}'.format(std), fontsize=10)
plt.show()
"""
"""
# Plot prediction
ax = plt.axes()
ax.plot(y,'r',label=Aprx)
ax.plot(Ypred,'b',label=Aprx+' Predicted')
ax.legend(loc='upper left')
ax.set_title('Prediction of '+Aprx+ '(train)')
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
ax.set_xlabel("Date and Year")
ax.set_ylabel("Market Value")
plt.show()
"""
"""
ConfutionMat=MovAvg.corr()# Correlation Amongst the Global Indices

# Visualization of Correlation matrix 
sns.heatmap(ConfutionMat,annot=True)
plt.xlabel('Indices')
plt.ylabel('Indices');
plt.title('Correlation Amongst Various Global Indices')
plt.show()
"""

"""
# Scatter plot to visulaize the correlation betwen two indices

Index='Nifty'
Clumn=MovAvg.columns
T=[i for i in range(len(Clumn)) if Clumn[i]!=Index]

fig, ax = plt.subplots(2, 4)
fig.subplots_adjust(hspace=0.5, wspace=0.6)
fig.suptitle('Correlation between '+Index+' and Other global Indices From June 08,2015 to June 08, 2020')
t=0 
for i in range(2):
    for j in range(4):
        ax[i,j].scatter(MovAvg.loc[:,Index],MovAvg.iloc[:,T[t]])
        ax[i,j].set(title=Index+' Vs '+Clumn[T[t]],xlabel=Index,ylabel=Clumn[T[t]])
        t+=1
    
plt.show()
"""
