#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv("data.csv")


# In[3]:


data


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.duplicated().sum()


# # DESCRIPTIVE ANALYSIS

# In[8]:


data["Age"]


# In[11]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.kdeplot(data["Age"],fill=True,color="lightgreen")
plt.title("KDE PLOT FOR AGE")
plt.subplot(1,2,2)
sns.histplot(data["Age"],bins=10,kde=False,color="cyan")
plt.title("HISTOGRAM PLOT FOR AGE")
plt.tight_layout()
plt.show()


# In[12]:


data["Age"].describe()

# INFERENCE
RANGE:20 TO 89 YEARS
MEAN AGE=54.42 YEARS
STANDARD DEVIATION=20.22
INTERQUATILE RANGE(IQR):37(Q1) TO 72(Q3)
THIS SUGGESTS A  BROAD REPRESENTATION OF BOTH YOUNG AND ELDERLY PATIENTS IN THE DATASET ,WHICH SUPPORTS AGE BASED COMPARATIVE ANALYSIS.
# In[13]:


data["Gender"].value_counts()


# In[15]:


data["Gender"].value_counts()
sns.barplot(x=data["Gender"].value_counts().index,
           y=data["Gender"].value_counts().values,
           palette=["blue","pink","green"])
for i,v in enumerate(data["Gender"].value_counts()):
    plt.text(i,v,str(v),ha="center",va="bottom")
plt.title("GENDER COUNT")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# INFERENCE:THIS DATASET CONTAINS THREE GENDER CATEGORIES(MALE,FEMALE,OTHER) WITH THE MOST COMMON BEING MALE(16796 RECORDS),GENDER DISTRIBUTION IS SUFFICIENT FOR EVALUATING GENDER SPECIFIC SURVIVAL TRENDS AND SEVERITY COLUMNS
# In[22]:


country_counts=data["Country_Region"].value_counts()
plt.figure(figsize=(5,5))
plt.pie(x=country_counts.values,
       labels=country_counts.index,
       autopct='%1.1f%%')
plt.title("COUNTRY/REGION DISTRIBUTION")
plt.show()


# In[23]:


country_counts

#INFERENCE:PATIENTS COME FROM 10 DIFFERENT COUNTRIES/REGIONS WITH AUSTRALIA BEING THE MOST REPRESENTED(5092 PATIENTS).
NUMBER OF DATA POINTS FOR EACH COUNTRY IS ALMOST SAME
THIS DIVERSITY ENABLES CROSS-COUNTRY COMPARISON OF CANCER OUTCOMES AND TREATMENT ECONOMIC
# In[35]:


# CANCER TYPE
data["Cancer_Type"].value_counts()


# In[37]:


data["Cancer_Type"].value_counts()
sns.barplot(x=data["Cancer_Type"].value_counts().index,
           y=data["Cancer_Type"].value_counts().values)
for i,v in enumerate(data["Cancer_Type"].value_counts()):
    
    plt.text(i,v,str(v),ha="center",va="bottom")
plt.title("Cancer type count")
plt.xlabel("Cancer Type")
plt.ylabel("Count")
plt.show()
    
    

# INFERENCE:WE HAVE IN TOTAL 8 TYPES OF CANCER WITH EACH CANCER HAVING APPROX SAME NUMBER OF DATA POINTS UNDER THE LABEL,MOST COMMON CANCERS ARE COLON CANCER FOLLOWED BY PROSTATE CANCER
# In[39]:


#CANCER STAGE
data["Cancer_Stage"].value_counts()
sns.barplot(x=data["Cancer_Stage"].value_counts().index,
           y=data["Cancer_Stage"].value_counts().values)
for i,v in enumerate(data["Cancer_Stage"].value_counts()):
    
    plt.text(i,v,str(v),ha="center",va="bottom")
plt.title("Cancer_Stage type count")
plt.xlabel("Cancer Type")
plt.ylabel("Count")
plt.show()
    
    

INFERENCE:CANCER STAGES HAVE 5 STAGES WITH VALUES RANGING FROM 0 TO4,WITH STAGE 2 THE MOST COMMON ONE,AND EACH STAGE HAVE THE ALMOST SAME NUMBER OF DATAPOINTS UNDER ITS LABEL 
# In[44]:


data["Treatment_Cost_USD"].describe()


# In[42]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.kdeplot(data["Treatment_Cost_USD"],fill=True,color="lightgreen")
plt.title("KDE PLOT FOR TREATMENT_COST")
plt.subplot(1,2,2)
sns.histplot(data["Treatment_Cost_USD"],bins=10,kde=False,color="cyan")
plt.title("HISTOGRAM PLOT FOR TREATMENT_COST")
plt.tight_layout()
plt.show()

INFERENCE:TREATMENT COST USD HAVE SAME SKEWNESS AND THERE ARE ALMOST SAME NUMBER OF DATA POINTS UNDER EACH BIN AS OBSEREVD BY HISTOGRAM
# # ANALYZING RISK FACTORS

# In[52]:


columns_of_interest=['Genetic_Risk',
       'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
summary=data[columns_of_interest].agg(["mean","min","max","std"])
summary

INFERENCE THESE VARIABLES HAVE NEARLY IDENTICAL MEANS AND STANDARD DEVIATION INDICATING THEY WERE LIKELY DESIGNED ON THE SAME STANDARIZED SCALE .
THEY ARE ESSENTIAL IN STUDYING THE INTERACTION EFECTS (EG:GENETIC RISK*SMOKING) ON SURVIVAL
# # 1: DETERMINE  THE RELATIONSHIP BETWEEN RISK FACTOR AND CANCER SEVERITY

# In[55]:


from scipy.stats import linregress


# In[70]:


risk_factors=['Genetic_Risk','Air_Pollution','Alcohol_Use','Smoking','Obesity_Level']
titles=['Genetic_Risk','Air_Pollution','Alcohol_Use','Smoking','Obesity_Level']
colors=['blue','green','orange','red','purple']
plt.figure(figsize=(20,12))
for i ,(factor,title,color) in enumerate(zip(risk_factors,titles,colors),1):
    plt.subplot(2,3,i)
    x=data[factor]
    y=data["Target_Severity_Score"]
    slope,intercept,r_value,p_value,std_error=linregress(x,y)
    r_squared=r_value**2
    sns.lineplot(x=factor,y="Target_Severity_Score",data=data,color=color)
    plt.plot(x,x*slope+intercept,color='black',linewidth=2,label="REGRESSION LINE")
    plt.ylabel("TARGET SEVERITY SCORE")
    plt.title(f"{title} VS SEVERITY SCORE\nR² = {r_squared:.2f}")
#line=y=mx+c

plt.tight_layout()
plt.show()
                        
                                        

# INFERENCE:To understand the contribution of various risk factors to cancer severity, line plots were generated for five
primary variables: 
Genetic Risk, Air Pollution, Alcohol Use, Smoking, and Obesity Level, plotted against the
Target Severity Score. All graphs reveal a positive relationship, indicating that as the level of a particular
risk factor increases, the corresponding severity of the condition also tends to rise. However, the degree
of association—measured by the slope and tightness of the confidence interval—varies across factors.
Genetic Risk vs Target Severity Score
R² = 0.23: A weak linear relationship. Only 23% of the variability in
Target_Severity_Score can be explained by Genetic_Risk. This suggests that other factors likely play a
larger role in influencing the severity score. Slope = 0.20: A positive slope indicates that as Genetic_Risk
increases, the Target_Severity_Score also tends to increase. For each unit increase in Genetic_Risk, the
target severity score increases by 0.20 units. However, because the R² is relatively low.
Air Pollution vs Target Severity Score 
R² = 0.13: A very weak relationship. Only 13% of the variance in
Target_Severity_Score can be explained by Air_Pollution, meaning that this factor has a limited effect on
the target variable. Slope = 0.15: A positive slope means that as air pollution increases, the severity score
slightly increases. But, due to the low R², this relationship is weak and unreliable as a predictor for the
target severity.
Alcohol Use vs Target Severity Score
R² = 0.13: Similarly, the relationship between Alcohol_Use and
Target_Severity_Score is also weak. Only 13% of the variation in the target score is explained by alcohol
use. Slope = 0.15: The positive slope indicates that increased alcohol use correlates with a slight increase
in target severity. However, like air pollution, the weak R² suggests other factors have a much stronger
influence on the target.
Smoking vs Target Severity Score
R² = 0.23: A weak relationship, similar to Genetic_Risk. Smoking explains
only 23% of the variance in the target score, leaving the majority of the variation to be explained by other
factors. Slope = 0.20: The positive slope implies that as smoking increases, the target severity score
increases as well. This relationship is similar to that of genetic risk, but with a weak linear association (low
R²).
Obesity Level vs Target Severity Score 
R²= 0.06: The weakest relationship among all factors. Only 6% of
the variation in the target score is explained by obesity level, suggesting that obesity has a minimal effect
on the target variable. 
Slope = 0.10: A positive slope, indicating a slight increase in the severity score as
obesity level increases. However, due to the very low R², this is a weak and unreliable relationship.

Key Takeaways:
Weak Linear Relationships: The R² values for all risk factors are relatively low, ranging from
0.06 to 0.23. This indicates that while there is some relationship between these risk factors and the
Target_Severity_Score, it is weak. These factors alone do not explain much of the variation in the target
variable.
Positive Trends:
All the slope values are positive, suggesting that as each risk factor increases, the
Target_Severity_Score tends to increase as well. However, because the R² values are low, this increase is
not strongly consistent across all data points.
Other Influences: The low R² values imply that other, unmeasured factors are likely contributing to the
variation in Target_Severity_Score. The risk factors you examined are only weakly correlated with the
target and are not reliable predictors on their own.
Next Steps:
Given the weak explanatory power of these individual factors, it might be useful to explore
other variables or more complex models that could account for more of the variation in the
Target_Severity_Score. This could include interactions between risk factors, adding new features, or
applying more sophisticated regression techniques.

# # ANALYZE THE PROPORTION OF EARLY STAGE DIAGNOSIS BY CANCER TYPE

# In[6]:


data["Cancer_Type"].unique()


# In[45]:


stage_count=data[data["Cancer_Type"]=="Lung"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of lungs cancer diagnosed at stage 0 and stage 1:  {proportion}")


# In[48]:


stage_count=data[data["Cancer_Type"]=="Leukemia"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of leukmania cancer diagnosed at stage 0 and stage 1: {proportion}")


# In[47]:


stage_count=data[data["Cancer_Type"]=="Breast"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of breastcancer diagnosed at stage 0 and stage 1: {proportion}")


# In[39]:


stage_count=data[data["Cancer_Type"]=="Colon"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of colon cancer diagnosed at stage 0 and stage 1: {proportion}")


# In[40]:


stage_count=data[data["Cancer_Type"]=="Skin"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of Skin cancer diagnosed at stage 0 and stage 1: {proportion}")


# In[41]:


stage_count=data[data["Cancer_Type"]=="Cervical"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of Cervical cancer diagnosed at stage 0 and stage 1: {proportion}")


# In[42]:


stage_count=data[data["Cancer_Type"]=="Prostate"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of colon Prostate diagnosed at stage 0 and stage 1: {proportion}")


# In[43]:


stage_count=data[data["Cancer_Type"]=="Liver"]["Cancer_Stage"].value_counts()
early_stage_sum=stage_count.get("Stage 0",0)+stage_count.get("Stage I",0)
total_sum=stage_count.sum()
proportion=(early_stage_sum/total_sum)*100
print(f"proportion of Liver cancer diagnosed at stage 0 and stage 1: {proportion}")

INFERENCE:
The analysis demonstrates that early-stage diagnosis for various cancer types is relatively widespread,with most cancers having an early diagnosis rate between 38.43%and 40.61%.Liver cancer shows the highest proportion while the lung cancer shows the lowest.These finding suggest that while screening and diagnostic methods are effective ,improvements can still be made,particularly in lung cancer detection.
Further research into screening strategies early intervention and use of the advanced diagnostic technologies could help increase the proportion of early stage diagnosis,ultimately leading to better surrvival rates and outcomes for cancer patient.
The relatively small variation across the cancer types indicates that in general,helathcare systems may need to focus on enhancing early detection uniformly with targeted efforts to address specific gaps in detection ,particularly for cancer like lung cancer,
# # IDENTIFY KEY PREDICTORS OF CANCER SEVERITY AND SURVIVAL YEARS

# In[56]:


features=['Age','Genetic_Risk','Air_Pollution','Alcohol_Use','Smoking','Obesity_Level']
targets=["Survival_Years","Target_Severity_Score"]
# calculate correlations
pearson_corr=data[features+targets].corr(method="pearson")
spearman_corr=data[features+targets].corr(method="spearman")
# slice out only the relationship between the target varialbles
pearson_result=pearson_corr[targets]
spearman_result=spearman_corr[targets]
# combine both
correlation_df=pd.concat([pearson_result,spearman_result],axis=1,keys=["Pearson","Spearman"])
correlation_df


# Key Findings:
# Smoking (0.484) and Genetic Risk (0.479) are the strongest predictors of disease severity.
# Air Pollution (0.367) and Alcohol Use (0.363) show moderate positive relationships with severity.
# Obesity Level (0.251) has a weaker but still noticeable effect.
# Age (-0.001) shows virtually no correlation with disease severity in this dataset.
# Pearson and Spearman values are very similar, suggesting that the relationships are generally monotonic and close to linear.

# In[88]:


# random forest for target severity score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
# converting categorical columns to numerical colums
categorical_cols=["Gender","Country_Region","Cancer_Type","Cancer_Stage"]
for col in categorical_cols:
    le=LabelEncoder()
    data[col]=le.fit_transform(data[col])
# preparing features and input
X=data.drop(columns=["Patient_ID","Survival_Years","Target_Severity_Score","Treatment_Cost_USD"])
y_severity=data["Target_Severity_Score"]
# train test split
X_train_s,X_test_s,y_train_s,y_test_s=train_test_split(X,y_severity,test_size=0.2,random_state=42)
# tarin model
model=RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=42)
# evaluate the model
model.fit(X_train_s,y_train_s)
train_r2_severity=r2_score(y_train_s,model.predict(X_train_s))
test_r2_severity=r2_score(y_test_s,model.predict(X_test_s))
print(train_r2_severity)
print(test_r2_severity)



# In[89]:


features_importance_severity=pd.Series(model.feature_importances_,index=X.columns,).sort_values(ascending=True)
# plotting of important features
plt.figure(figsize=(10,6))
features_importance_severity.plot(kind="bar",color="skyblue")
plt.title("Feature importance for target severity score(Random Forest)")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# random forest for target severity score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
# converting categorical columns to numerical colums
categorical_cols=["Gender","Country_Region","Cancer_Type","Cancer_Stage"]
for col in categorical_cols:
    le=LabelEncoder()
    data[col]=le.fit_transform(data[col])
# preparing features and input
X=data.drop(columns=["Patient_ID","Survival_Years","Target_Severity_Score","Treatment_Cost_USD"])
y_severity=data["Survival_Years"]
# train test split
X_train_s,X_test_s,y_train_s,y_test_s=train_test_split(X,y_severity,test_size=0.2,random_state=40)
param_grid={
"n_estimators":[100,200],
"max_depth":[5,10,None],
"min_samples_split":[2,5],
"min_samples_leaf":[1,2]}
# tarin model
model=RandomForestRegressor(random_state=40)
GSC=GridSearchCV(model,param_grid,cv=3,scoring="r2",n_jobs=1)

# evaluate the model
GSC.fit(X_train_s,y_train_s)
best_rf_severity=GSC.best_estimator_
train_r2_severity=r2_score(y_train_s,best_rf_severity.predict(X_train_s))
test_r2_severity=r2_score(y_test_s,best_rf_severity.predict(X_test_s))
print(train_r2_severity)
print(test_r2_severity)


# In[3]:


# random forest for target severity score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
# converting categorical columns to numerical colums
categorical_cols=["Gender","Country_Region","Cancer_Type","Cancer_Stage"]
for col in categorical_cols:
    le=LabelEncoder()
    data[col]=le.fit_transform(data[col])
# preparing features and input
X=data.drop(columns=["Patient_ID","Survival_Years","Target_Severity_Score","Treatment_Cost_USD"])
y_severity=data["Survival_Years"]
# train test split
X_train_s,X_test_s,y_train_s,y_test_s=train_test_split(X,y_severity,test_size=0.2,random_state=42)
# tarin model
model=RandomForestRegressor(n_estimators=200,max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=42)
# evaluate the model
model.fit(X_train_s,y_train_s)
train_r2_severity=r2_score(y_train_s,model.predict(X_train_s))
test_r2_severity=r2_score(y_test_s,model.predict(X_test_s))
print(train_r2_severity)
print(test_r2_severity)



# In[5]:


import seaborn as sns
sns.histplot(data["Survival_Years"],kde=True)


# In[6]:


data.corr(numeric_only=True)["Survival_Years"].sort_values(ascending=True)


# # EXPLORE THE ECONOMIC BURDEN OF CANCER TREATMENT ACROSS DIFFERENT DEMOGRAPHICS AND COUNTRIES

# In[11]:


data


# In[12]:


df=data.copy()


# In[ ]:





# In[15]:


df


# In[19]:


country_gender_cost


# In[22]:


data["Age_Group"]=pd.cut(data["Age"],bins=[0,30,45,60,75,100],labels=["0-30","31-45","46-60","61-75","76+"])


# In[23]:


data


# In[31]:


country_Age_Cost=data.groupby(["Country_Region","Age_Group","Gender"])["Treatment_Cost_USD"].mean().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(data=country_Age_Cost,x="Country_Region",y="Treatment_Cost_USD",hue="Gender")
plt.title("Average cancer treatment cost by country and gender")
plt.show()


# In[36]:


heatmap_data=country_Age_Cost.pivot_table(
    index="Age_Group",
    columns="Country_Region",
    values="Treatment_Cost_USD",
    aggfunc="mean"
)


# In[37]:


plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data,annot=True,fmt=".0f")
plt.title("Average Treatment_Cost by Age_Group & Country")


# Demographic Inferences
# 1. Healthcare spending does not appear to be strongly age-dependent
# 
# Across most countries, treatment costs remain within a narrow range across age groups. The difference between the youngest (0–30) and oldest (76+) groups is relatively small compared to the overall cost level.
# 
# Business Insight: Age alone may not be a strong predictor of treatment expenditure. Other factors such as disease type, healthcare system, insurance coverage, or treatment complexity may have greater influence.
# 
# 2. Elderly populations tend to have slightly higher treatment costs
# 
# In countries such as Australia, China, India, and the USA, the 76+ age group generally shows higher average treatment costs than younger groups.
# 
# Demographic Insight: As populations age, healthcare systems may experience increased financial pressure due to higher utilization of medical services by senior citizens.
# 
# 3. Cross-country variation is greater than age-group variation
# 
# For example, the same age group can show noticeable cost differences between countries, while differences between age groups within a single country are often modest.
# 
# Key Finding: Geographic location has a stronger impact on treatment costs than age demographics in this dataset.
# 
# 4. Middle-aged groups (31–60) do not consistently incur the highest costs
# 
# Contrary to expectations that working-age adults may have higher healthcare expenditures, the heatmap does not show a consistent peak among the 31–45 or 46–60 groups.
# 
# Interpretation: Healthcare expenditure is likely distributed across all age groups rather than concentrated in a specific demographic segment.
# 
# 5. Australia exhibits consistently higher healthcare costs across demographics
# 
# Australia records some of the highest average treatment costs for multiple age groups, particularly among the elderly.
# 
# Possible Interpretation: This may reflect higher healthcare prices, greater access to advanced treatments, or differences in healthcare delivery systems.
# 
# 6. Pakistan and India show comparatively lower treatment costs
# 
# Several age groups in these countries have lower average costs than their counterparts in developed nations.
# 
# Demographic Implication: Lower healthcare expenditure may be associated with lower treatment costs, different healthcare pricing structures, or variations in healthcare accessibility.

# # ASSESS WHETHER HIGHER TREATMENT COST IS ASSCOIATED WITH LONGER SURVIVAL
Null Hypothesis(Ho):-There is no corelation between treatment cost and survival years.
Alternative Hypothesis(H1):-There is a correlation between treatment cost and survival years.
# In[44]:


from scipy.stats import pearsonr,spearmanr
x=data["Treatment_Cost_USD"]
y=data["Survival_Years"]
# performing pearson correlation test
pearson_corr,pearson_p=pearsonr(x,y)
print(f"Pearson correlation coefficient'{pearson_corr}")
print(f"Pearson P-value:{pearson_p}")

#spearman correlation test
spearman_corr,spearman_p=spearmanr(x,y)
print(f"spearman correlation coefficient'{spearman_corr}")
print(f"spearman P-value:{spearman_p}")
alpha=0.05
def interpret_corr(corr,p,method):
    if p<alpha:
        print(f"{method},we reject the null hypothesis")
    else:
        print(f"{method},we failed to reject the null hypothesis")
interpret_corr(pearson_corr,pearson_p,"Pearson")
interpret_corr(pearson_corr,spearman_p,"Spearman")


# In[46]:


sns.regplot(x=x,y=y,line_kws={"color":"red"})

 THERE IS NO RELATIONSHIP BETWEEN TREATMENT COST AND SURVIVAL YEARS
# # EVALUATE IF HIGHER CANCER STAGES LEAD TO GREATER TREATMENT COSTS AND REDUCES SURVIVAL YEARS

# In[12]:


df=data.copy()


# In[13]:


stage_order=['Stage 0','Stage I','Stage II',"Stage III","Stage IV"]


# In[14]:


grouped_stats=df.groupby("Cancer_Stage")[["Treatment_Cost_USD","Survival_Years"]].mean().reset_index()


# In[15]:


grouped_stats

TREATMENT COST VS CANCER STAGE
NULL HYPOTHESIS(Ho):The average treatment cost is the same across all stages
Alternative Hypothesis(H1):At least one stage has differnt average cost
Survival Years VS Cancer Stage
Null Hypothesis(Ho):The average survival years are the same across all cancer stages
Alternative Hypothesis:At least one stage has a different survival duration
# In[20]:


grouped_cost=[]
grouped_survival=[]
for stage in stage_order:
    stage_data=df[df["Cancer_Stage"]==stage]
    cost=stage_data["Treatment_Cost_USD"]
    survival=stage_data["Survival_Years"]
    grouped_cost.append(cost)
    grouped_survival.append(survival)

check the normality
H0:all groups are normally distributed
H1:all groups are not normally distributed
# In[18]:


# check normality
from scipy.stats import shapiro,f_oneway
normal_cost=0
normal_survival=0
for i in range (len(stage_order)):
    cost_p=shapiro(grouped_cost[i]).pvalue
    survival_p=shapiro(grouped_survival[i]).pvalue
    print(f"{cost_p} for group{i}")
    print(f"{survival_p} for group{i}")
    if cost_p<0.05:
          normal_cost+=1
    if survival_p<0.05:
        
        normal_survival+=1
print(cost_p)
print(survival_p)
          
    


# In[26]:


from scipy.stats import shapiro, f_oneway

normal_cost = 0
normal_survival = 0

for i in range(len(stage_order)):
    cost_p = shapiro(grouped_cost[i]).pvalue
    survival_p = shapiro(grouped_survival[i]).pvalue

    print(f"Cost p-value: {cost_p} for group {i}")
    print(f"Survival p-value: {survival_p} for group {i}")

    if cost_p <= 0.05:
        normal_cost += 1
    if survival_p <= 0.05:
        normal_survival += 1

print("\nSummary:")
print("Normal cost groups:", normal_cost)
print("Normal survival groups:", normal_survival)


# In[24]:


print(normal_cost)


# In[29]:


from scipy.stats import kruskal
kruskal_cost=kruskal(*grouped_cost)
kruskal_survival=kruskal(*grouped_survival)
p_cost=kruskal_cost.pvalue
p_survival=kruskal_survival.pvalue


# In[30]:


p_cost


# In[31]:


p_survival

conclusion:
KRUSKAL -WALLIS TEST:TREATMENT COST ACROSS CANCER STAGES===
P_VALUE=0.42544
CONCLUSION:NO SIGNIFICANT DIFFERENCE IN TREATMENT COST AMONG CANCER STAGES
=== KRUSKAL WALLIS TEST:SURVIVAL YEARS ACROSS CANCER STAGES
P_VALUE=0.6033
CONCLUSION:THERE IS NO SIGNIFICANT DIFFERENCE IN SURVIVAL_YEARS AMONG CANCER_STAGES
# # EXAMINE WHETHER HIGHER GENETIC RISK AMPLIFIES THE NEGATIVE EFFECTS OF SMOKING ON CANCER SEVERITY AND SURVIVAL OUTCOMES
NULL HYPOTHESIS(Ho):
    THE INTERACTION EFFECT BETWEEN GENETIC_RISK AND SMOKING ON CANCER SEVERITY IS NOT SIGNIFICANT
    (GENETICS RISK DOES NOT AMPLIFY OR ALTER THE EFFECTS OF SMOKING)
ALTERNATIVE HYPOTHESIS(H1):
    THE INTERACTION EFFECT BETWEEN GENETIC_RISK AND SMOKING ON CANCER SEVERITY IS SIGNIFICANT
    (GENETICS RISK AMPLIPY OR ALTER THE EFFECTS OF SMOKING)
    
# In[43]:


import statsmodels.formula.api as smf

model = smf.ols(
    "Target_Severity_Score ~ Genetic_Risk * Smoking",
    data=data
).fit()

model.summary2().tables[1].loc["Genetic_Risk:Smoking"]

P_VALUE=0.628255
Conclusion

There is no statistically significant interaction effect between Genetic Risk and Smoking on Cancer Severity (p = 0.628). The results suggest that the impact of smoking on cancer severity does not significantly change across different levels of genetic risk. In other words, genetic risk does not significantly amplify or alter the effect of smoking on the target severity score in this dataset.

# In[ ]:




