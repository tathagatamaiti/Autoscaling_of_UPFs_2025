import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
updated_data = pd.read_csv('cleaned_data.csv')

# Plotting PDU_session vs averaged Time_instance showcasing averaged CPU_share
plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    updated_data['Time_instance'],
    updated_data['PDU_session'],
    c=updated_data['CPU_share'],
    cmap='plasma',
    alpha=0.7
)

# Adding color bar and setting font sizes
cbar = plt.colorbar(scatter)
cbar.set_label('CPU Share', fontsize=16)
cbar.ax.tick_params(labelsize=14)
plt.xlabel('Averaged Time Instance', fontsize=16)
plt.ylabel('PDU Session', fontsize=16)
plt.title('PDU Sessions over Averaged Time with CPU Share Intensity', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

# Annotate each point with the UPF anchoring information
for i, row in updated_data.iterrows():
    plt.text(
        row['Time_instance'],
        row['PDU_session'],
        f"{row['UPF_instance']}",
        fontsize=14,
        ha='right',
        va='bottom',
        alpha=0.7
    )

#Save plot
plt.savefig('PDU_Time_CPUshare.png', format='png', dpi=300)

