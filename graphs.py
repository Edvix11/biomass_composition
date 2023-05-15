import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel('grp.xlsx')
df_g = pd.read_csv('g_values.csv')

NH4_bound = df['nh4_bound'].tolist()
ATP = df['atp_c'].tolist()
H2O = df['h2o_c'].tolist()
NAD = df['nad_c'].tolist()
Glutamate = df['glu__L_c'].tolist()
aCoA = df['accoa_c'].tolist()
pe161 = df['pe161_c'].tolist()

H4_bound_g = df['nh4_bound'].tolist()
ATP_g = df_g['atp_c'].tolist()
H2O_g = df_g['h2o_c'].tolist()
NAD_g = df_g['nad_c'].tolist()
Glutamate_g = df_g['glu__L_c'].tolist()
aCoA_g = df_g['accoa_c'].tolist()
pe161_g = df_g['pe161_c'].tolist()

ATP_BOF = 54.119975
H2O_BOF = 48.752916
NAD_BOF = 0.001787
Glutamate_BOF = 0.255712
aCoA_BOF = 0.000279
pe161_BOF = 0.009618

metabolites = [ 
               ('ATP', ATP, ATP_g, ATP_BOF), 
               ('H2O', H2O, H2O_g, H2O_BOF), 
               ('NAD', NAD, NAD_g, NAD_BOF), 
               ('Glutamate', Glutamate, Glutamate_g, Glutamate_BOF),
               ('Acetyl-CoA', aCoA, aCoA_g, aCoA_BOF), 
               ('Phosphatidylethanolamine', pe161, pe161_g, pe161_BOF)]

#%%
fig, axs = plt.subplots(3, 2, figsize=(13, 13))  # Create a figure with 6 subplots
axs = axs.ravel()  # Flatten the axes array

# Iterate over the metabolites
for i, (metabolite, data, data_g, BOF) in enumerate(metabolites):
    # Plot the data on the i-th subplot
    axs[i].plot(NH4_bound, data, label='ME-model predicted')
    axs[i].plot(H4_bound_g, data_g, linestyle='--', label='ML-predicted')
    axs[i].axhline(y=BOF, color='r', label='iJO1366 BOF coefficient')

    axs[i].set_xlabel('Ammonium lower bound')
    axs[i].set_ylabel(metabolite)
    axs[i].set_title(f'{metabolite} flux')

    axs[i].legend()
    if metabolite == "Phosphatidylethanolamine":
        axs[i].set_ylim(bottom=0, top = BOF*100)
    else:
        axs[i].set_ylim(bottom=0, top = BOF*2)


plt.tight_layout()
plt.show()



