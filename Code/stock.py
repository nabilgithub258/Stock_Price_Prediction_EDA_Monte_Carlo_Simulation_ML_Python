#!/usr/bin/env python
# coding: utf-8

# In[2]:


#####################################################################################################
######################### STOCK DATA SET  ###########################################################
#####################################################################################################


# In[3]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


##########################################################################
########### Getting one year data from yahoo finance stock
##########################################################################

import yfinance as yf

# Define the ticker symbols
stocks = ['AAPL', 'GOOG','MSFT','AMZN']

# Download stock data

for i in stocks:
    globals()[i] = yf.download(i, start='2023-01-01', end='2024-01-01')
    
#### use the globals if you wanna make a seperate data set for each of the stock, much better and efficient


# In[5]:


###################################################################
####################### Part II - EDA
###################################################################

AAPL.head()


# In[6]:


GOOG.head()


# In[7]:


AAPL.describe()


# In[8]:


AAPL.info()


# In[9]:


AAPL['Adj Close'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10,linestyle='dashed',linewidth=2)

plt.title('AAPL Stock Adj Close Graph')

plt.xlabel('Date')

plt.ylabel('Adj Close')

#### clearly we see a rise in the stock as the dates go further


# In[10]:


AAPL.Volume.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10,linestyle='dashed',linewidth=3)


# In[11]:


#### now lets make the moving average

ma_days = [10,20,30,40,50]

for i in ma_days:
    column_name = f'MA for {i} days'
    
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=i).mean()


# In[12]:


AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 30 days','MA for 40 days','MA for 50 days']].plot(legend=True,figsize=(20,7))

#### moving averages for 10,20,30,40,50 days


# In[13]:


#### if you find for loop method complicated then you can also do single column by column like this

AAPL['MA for 60 days'] = AAPL['Adj Close'].rolling(window=60).mean()


# In[14]:


AAPL.head()


# In[15]:


AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 30 days','MA for 40 days','MA for 50 days','MA for 60 days']].plot(legend=True,figsize=(20,7))

#### updated version


# In[16]:


#### now we will calculate the percentage change or pct change

AAPL['Daily change'] = AAPL['Adj Close'].pct_change()


# In[17]:


AAPL.head()


# In[18]:


AAPL['Daily change'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10,linestyle='dashed',linewidth=3)

#### this way we can see from the start if the change in stock was profitable or not


# In[19]:


#### now lets make the histogram of the new column we just created, instead of doing simple histogram we can do dist plot

sns.displot(x='Daily change',data=AAPL,rug=True,bins=50,kde=True,aspect=2,height=7,stat='density',color='black')


# In[20]:


g = sns.jointplot(x='Adj Close',y='Daily change',data=AAPL,kind='kde',fill=True,color='red')

g.fig.set_size_inches(17,9)

#### the density seems to fall under 180 and 0 on y axis which is interesting but also makes sense in long term


# In[21]:


g = sns.jointplot(x='Adj Close',y='Daily change',data=AAPL,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

#### we dont see much honestly


# In[22]:


AAPL['Daily change'].isnull().any()


# In[23]:


AAPL['Daily change'] = AAPL['Daily change'].fillna(0)

AAPL.head()


# In[24]:


AAPL['Daily change'].isnull().any()


# In[25]:


from scipy.stats import pearsonr                  #### lets see this with pearsonr


# In[26]:


co_eff, p_value = pearsonr(AAPL['Adj Close'],AAPL['Daily change'])


# In[27]:


co_eff                              #### bad news


# In[28]:


p_value                             #### definately not correlated


# In[29]:


closing_df = yf.download(stocks,start='2023-01-01', end='2024-01-01')['Adj Close']


# In[30]:


#### now we have the all 4 stocks adj closing in one single data frame called closing_df

closing_df.head()


# In[31]:


return_df = closing_df.pct_change().fillna(0)


# In[32]:


#### now we have the return change meaning the percentage of change after closing

return_df.head()


# In[33]:


return_df['AMZN'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)

plt.title('Amazon Return Profit Graph')

plt.xlabel('Date')

plt.ylabel('Profit Density')

#### this is quite interesting, if you invested in AMZN stock at the start of 2023 and withdrew at the start of 2024 you would be in loss
#### now lets see the worst and best days for the amazon stock


# In[34]:


return_df[return_df.AMZN == return_df.AMZN.min()]


# In[35]:


return_df['AMZN'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)

plt.title('Amazon Return Profit Graph')

plt.xlabel('Date')

plt.ylabel('Profit Density')

plt.axvline(x='2023-02-03',linewidth=3,color='red')

#### the date 2023-02-03 was the worst day for amazon stock


# In[36]:


return_df[return_df.AMZN == return_df.AMZN.max()]


# In[37]:


return_df['AMZN'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)

plt.title('Amazon Return Profit Graph')

plt.xlabel('Date')

plt.ylabel('Profit Density')

plt.axvline(x='2023-08-04',linewidth=3,color='red')

#### the best day being 2023-08-04 for amazon stocks


# In[38]:


avg = return_df[['AAPL','AMZN','GOOG','MSFT']].mean()

avg = pd.DataFrame(avg)


# In[39]:


avg                      #### it seems like AMZN was the most profitable when it comes down to the mean


# In[40]:


avg.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',linestyle='dashed',linewidth=4,markersize=20)

plt.scatter(avg.index[0], avg.iloc[0], color='white', s=150, zorder=5)

plt.scatter(avg.index[1], avg.iloc[1], color='brown', s=150, zorder=5)

plt.scatter(avg.index[2], avg.iloc[2], color='green', s=150, zorder=5)

plt.scatter(avg.index[3], avg.iloc[3], color='blue', s=150, zorder=5)

plt.title('Stock Mean Profit Graph')

plt.xlabel('Company')

plt.ylabel('Density Mean')

#### the least profitable being Apple which is quite shocking to say the least and the most profitable being amazon


# In[41]:


std = return_df[['AAPL','AMZN','GOOG','MSFT']].std()

std = pd.DataFrame(std)

std.columns = ['STD']

std


# In[42]:


std.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',linestyle='dashed',linewidth=4,markersize=20)

plt.scatter(std.index[0], std.iloc[0], color='white', s=150, zorder=5)

plt.scatter(std.index[1], std.iloc[1], color='brown', s=150, zorder=5)

plt.scatter(std.index[2], std.iloc[2], color='green', s=150, zorder=5)

plt.scatter(std.index[3], std.iloc[3], color='blue', s=150, zorder=5)

plt.title('Stock STD Graph')

plt.xlabel('Company')

plt.ylabel('Density STD')

#### ideally you wanna invest in stocks whos STD is low and is safer to invest, although Amazon stock yielded more profit
#### but it has a higher STD making it quite risky to invest
#### apple is most safest stock here


# In[43]:


return_df.head()


# In[44]:


g = sns.jointplot(x='AAPL',y='MSFT',data=return_df,kind='kde',fill=True,color='black')

g.fig.set_size_inches(17,9)

#### interesting


# In[45]:


#### lets see if they are corelated with linear regression plot 

g = sns.lmplot(x='AAPL',y='MSFT',data=return_df,scatter_kws={'color':'black'},line_kws={'color':'red'})

g.fig.set_size_inches(17,9)

#### clearly we as the stocks for AAPL goes up, the similar effect takes place to MSFT stocks
#### lets the correlation between them


# In[46]:


co_eff,p_value = pearsonr(return_df.AAPL,return_df.MSFT)

co_eff                        #### looking good, definately correlated


# In[47]:


p_value                       #### we reject null hypothesis


# In[48]:


corr = return_df.corr()

corr


# In[49]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### in short they all are correlated to each to some degree but if you dig deep then you will see which one is highly correlated and which ones are not
#### lets see GOOG, this one we have not given any attention so lets dive deeper on this one
#### GOOG and AMZN stands out here from others


# In[50]:


return_df['GOOG'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='blue',color='black',markersize=10)

return_df['MSFT'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black',markersize=10)

return_df['AAPL'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='white',color='black',markersize=10)

return_df['AMZN'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='brown',color='black',markersize=10)


plt.title('Return Profit Graph')

plt.xlabel('Date')

plt.ylabel('Profit Density')

#### seems like when one goes down the other one also somehow goes down which is basically a correlation, no wonder they are so closely correlated


# In[51]:


g = sns.lmplot(x='AMZN',y='GOOG',data=return_df,scatter_kws={'color':'black'},line_kws={'color':'red'})

g.fig.set_size_inches(17,9)

#### very much correlated, amazing
#### apart from all the companies in our df, these are the most correlated to each other then any


# In[52]:


return_df.head()


# In[53]:


df = return_df.reset_index()

df


# In[54]:


df.info()


# In[55]:


df['month'] = df.Date.apply(lambda x:x.month)

df.month.head()


# In[56]:


df['month_name'] = df.month.map({1:'Jan',
                         2:'Feb',
                         3:'Mar',
                         4:'Apr',
                         5:'May',
                         6:'Jun',
                         7:'Jul',
                         8:'Aug',
                         9:'Sep',
                         10:'Oct',
                         11:'Nov',
                         12:'Dec'})


# In[57]:


df['day_of_week'] = df.Date.apply(lambda x:x.dayofweek)

df['Day'] = df.day_of_week.map({0:'Mon',
                                     1:'Tue',
                                     2:'Wed',
                                     3:'Thr',
                                     4:'Fri',
                                     5:'Sat',
                                     6:'Sun'})

df.head()


# In[58]:


df['day'] = df.Date.apply(lambda x:x.day)

df.head()


# In[59]:


heat = df.groupby(['month_name','Day','day'])['AAPL'].sum().unstack().unstack().fillna(0)

heat


# In[60]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### this summarizes the whole Apple stock in one heatmap for one whole year


# In[61]:


heat['sum'] = heat.sum(axis=1)             #### we wanna see which months were most profitable 

heat.head()


# In[62]:


heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2             #### Jan being the best month for profit while Sep being the worst for AAPL


# In[63]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### same we see inside the heatmap for AAPL


# In[64]:


heat = df.groupby(['Day','day'])['AAPL'].sum().unstack().fillna(0)

heat


# In[65]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### seems like friday is doing well but we wanna be sure about it


# In[66]:


heat['sum'] = heat.sum(axis=1)             #### we wanna see which days were most profitable 

heat


# In[67]:


heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[68]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### the most profitable day being Friday and worst being Wednesday for AAPL stocks


# In[69]:


heat = df.groupby(['month_name','Day','day'])['GOOG'].sum().unstack().unstack().fillna(0)

heat        #### similar treatment to Google stocks


# In[70]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### this summarizes the whole Apple stock in one heatmap for one whole year


# In[71]:


heat['sum'] = heat.sum(axis=1)             #### we wanna see which months were most profitable 

heat.head()


# In[72]:


heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2             #### April being the best month for profit while Feb being the worst for GOOG


# In[73]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### same we see inside the heatmap for GOOG


# In[74]:


heat = df.groupby(['Day','day'])['GOOG'].sum().unstack().fillna(0)

heat


# In[75]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### seems like friday is doing well but we wanna be sure about it


# In[76]:


heat['sum'] = heat.sum(axis=1)             #### we wanna see which days were most profitable 

heat


# In[77]:


heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[78]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### the most profitable day being Thursday and worst being Wednesday for GOOG stocks, so Wednesday is worst for both AAPL and GOOG, interesting


# In[79]:


heat = df.groupby(['month_name','Day','day'])[['AAPL','AMZN','MSFT','GOOG']].sum().unstack().unstack().fillna(0)

heat                 #### eveything in one


# In[80]:


fig, ax = plt.subplots(figsize=(100,52))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')


# In[81]:


heat['sum'] = heat.sum(axis=1)             #### this will be interesting

heat    


# In[82]:


heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2                        #### March being the best month for all stocks, interesting


# In[83]:


fig, ax = plt.subplots(figsize=(100,52))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### this has all the information you will need honestly


# In[84]:


heat = df.groupby(['Day','day'])[['AAPL','AMZN','MSFT','GOOG']].sum().unstack().fillna(0)

heat['sum'] = heat.sum(axis=1) 

heat


# In[85]:


heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[86]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### similar thing we here, Friday and Thursday being the best days to trade while Wednesday being the worst as a customer


# In[87]:


#### we will now see the Risk Analysis

return_df.head()


# In[88]:


risk = return_df


# In[89]:


fig, ax = plt.subplots(figsize=(15,4))

pl = sns.scatterplot(x=risk.mean(),y=risk.std(),ax=ax,color='black')

pl.set_xlabel('Expected Returns')

pl.set_ylabel('Risk')

ax.set_xlim(risk.mean().min() - 0.0005, risk.mean().max() + 0.0005)

ax.set_ylim(risk.std().min() - 0.0005, risk.std().max() + 0.0005)

ax.set_xticks([round(x, 5) for x in plt.xticks()[0]])

ax.set_yticks([round(y, 5) for y in plt.yticks()[0]])


for i,ticker in enumerate(risk.mean().index):
    pl.text(risk.mean()[i],
           risk.std()[i],ticker,fontsize=9,ha='right')

#### just like we had previously observed, AAPL being the most safe and reliable investment, while AMZN being the most risky


# In[90]:


fig, ax = plt.subplots(figsize=(15, 4))

colors = ['black', 'brown', 'blue', 'green']

for i, ticker in enumerate(risk.mean().index):
    sns.scatterplot(x=[risk.mean()[ticker]], y=[risk.std()[ticker]], ax=ax, color=colors[i], label=ticker)

plt.xlabel('Expected Returns')
plt.ylabel('Risk')

ax.set_xlim(risk.mean().min() - 0.0005, risk.mean().max() + 0.0005)
ax.set_ylim(risk.std().min() - 0.0005, risk.std().max() + 0.0005)
ax.set_xticks([round(x, 5) for x in plt.xticks()[0]])
ax.set_yticks([round(y, 5) for y in plt.yticks()[0]])

for i, ticker in enumerate(risk.mean().index):
    plt.text(risk.mean()[i], risk.std()[i], ticker, fontsize=9, ha='right')

plt.legend()
plt.show()

#### same plot but we wanted to use different colors for different stocks, nothing fancy just color change


# In[91]:


return_df.head()


# In[93]:


return_df.AAPL.quantile(0.05)   #### what this means is that with 95% confidence we can say our one day loss will not exceed 1.7%

#### or in other words 95% of the data set is above this value


# In[94]:


return_df.AAPL.quantile(0.95)   #### this means its the 95 percentile of the data set, meaning only 5% is below this


# In[95]:


return_df.AMZN.quantile(0.05)   #### 95% confidence and one day loss at any day not exceed 2.7% for Amazon


# In[96]:


return_df.GOOG.quantile(0.05)   #### 95% confidence, one day loss at any day not exceed 2.6% for Google


# In[97]:


return_df.MSFT.quantile(0.05)   #### 95% confidence, one day loss at any day not exceed 2.2% for Microsoft


# In[98]:


return_df.GOOG.mean()


# In[99]:


return_df.info()


# In[100]:


return_df.GOOG.quantile(0.01)    #### 99% confidence level, worst one day loss cant exceed 4.2% at any given day of Google stock


# In[101]:


#### monte carlo method to predict the future stock price

closing_df


# In[102]:



mean_return = return_df.GOOG.mean()
std_return = return_df.GOOG.std()

closing_prices = closing_df.GOOG


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Google Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')

#### black line is our closing price and the rest is the predictions, you will realize the majority of data points increase in stock price as the days grows bigger
#### while lower band suggests that the price will go down, again here we see the percentile difference


# In[103]:


final_prices = simulated_prices[:, -1]

plt.figure(figsize=(15, 6))
sns.histplot(final_prices, bins=100, alpha=0.7, edgecolor='black',color='blue')

quantiles = [0.05, 0.5, 0.95]
quantile_values = np.quantile(final_prices, quantiles)

for quantile, value in zip(quantiles, quantile_values):
    plt.axvline(x=value, color='red', linestyle='--')
    plt.text(value, plt.ylim()[1] * 0.9, f'{int(quantile*100)}th: ${value:.2f}', color='red')

# Title and labels
plt.title(f'Histogram of Simulated Final Prices for Google')
plt.xlabel('Price')
plt.ylabel('Frequency')


# In[104]:


closing_prices[-1]


# In[106]:


#### because amazon was so highly correlated to google stock so now lets do the same treatment to amazon stocks

mean_return = return_df.AMZN.mean()
std_return = return_df.AMZN.std()

closing_prices = closing_df.AMZN


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 10000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Amazon Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[108]:


final_prices = simulated_prices[:, -1]

plt.figure(figsize=(15, 6))
sns.histplot(final_prices, bins=100, alpha=0.7, edgecolor='black',color='brown')

quantiles = [0.05, 0.5, 0.95]
quantile_values = np.quantile(final_prices, quantiles)

for quantile, value in zip(quantiles, quantile_values):
    plt.axvline(x=value, color='red', linestyle='--')
    plt.text(value, plt.ylim()[1] * 0.9, f'{int(quantile*100)}th: ${value:.2f}', color='red')

# Title and labels
plt.title(f'Histogram of Simulated Final Prices for Amazon')
plt.xlabel('Price')
plt.ylabel('Frequency')


# In[109]:


closing_prices[-1]


# In[114]:


mean_return = return_df.AAPL.mean()
std_return = return_df.AAPL.std()

closing_prices = closing_df.AAPL


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 100000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Apple Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')

#### according to our research and our limited data points at this point, AAPL seems to be most sensible one to invest


# In[115]:


final_prices = simulated_prices[:, -1]

plt.figure(figsize=(15, 6))
sns.histplot(final_prices, bins=100, alpha=0.7, edgecolor='black',color='white')

quantiles = [0.05, 0.5, 0.95]
quantile_values = np.quantile(final_prices, quantiles)

for quantile, value in zip(quantiles, quantile_values):
    plt.axvline(x=value, color='red', linestyle='--')
    plt.text(value, plt.ylim()[1] * 0.9, f'{int(quantile*100)}th: ${value:.2f}', color='red')

# Title and labels
plt.title(f'Histogram of Simulated Final Prices for Apple')
plt.xlabel('Price')
plt.ylabel('Frequency')

#### the safest stock to invest in from this alone


# In[116]:


closing_prices[-1]


# In[117]:


mean_return = return_df.MSFT.mean()
std_return = return_df.MSFT.std()

closing_prices = closing_df.MSFT


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 100000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Microsoft Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[119]:


final_prices = simulated_prices[:, -1]

plt.figure(figsize=(15, 6))
sns.histplot(final_prices, bins=100, alpha=0.7, edgecolor='black',color='green')

quantiles = [0.05, 0.5, 0.95]
quantile_values = np.quantile(final_prices, quantiles)

for quantile, value in zip(quantiles, quantile_values):
    plt.axvline(x=value, color='red', linestyle='--')
    plt.text(value, plt.ylim()[1] * 0.9, f'{int(quantile*100)}th: ${value:.2f}', color='red')

# Title and labels
plt.title(f'Histogram of Simulated Final Prices for Microsoft')
plt.xlabel('Price')
plt.ylabel('Frequency')


# In[118]:


closing_prices[-1]


# In[ ]:


############################################################################################################################
#### In this stock data analysis project, we analyzed data for GOOG, AMZN, AAPL, and MSFT. We calculated and ###############
#### visualized moving averages, then analyzed return profit using percentage change on closing prices. Several ############
#### plots revealed the highest correlation between GOOG and AMZN. Finally, we employed Monte Carlo simulation to ##########
#### predict future stock prices for all the analyzed stocks. ##############################################################
############################################################################################################################

