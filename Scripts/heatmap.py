import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
offline_solution = pd.read_csv('offline_solution.csv')

# Group by PDU_session and UPF_instance, then calculate the average CPU share
cpu_share_data = offline_solution[['PDU_session', 'UPF_instance', 'CPU_share']]
heatmap_data = cpu_share_data.pivot_table(index='PDU_session', columns='UPF_instance', values='CPU_share', aggfunc='mean').fillna(0)

# Plot heatmap
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'CPU Share'}, annot_kws={"size": 15})
plt.title("Heatmap of CPU Share Allocation across UPFs and PDUs", fontsize=25)
plt.xlabel("UPF Instance", fontsize=20)
plt.ylabel("PDU Session", fontsize=20)

#Save plot
plt.savefig('PDU_UPF_heatmap.png', format='png')