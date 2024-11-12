import pandas as pd
import matplotlib.pyplot as plt

# Load the offline solution data
offline_solution = pd.read_csv('offline_solution.csv')

# Filter for only active UPFs
active_upfs = offline_solution[offline_solution['UPF_active'] == 1]

# Calculate active UPFs per time instance
upf_usage = active_upfs.groupby('Time_instance')['UPF_instance'].nunique()

# Calculate the average UPF usage
average_upf_usage = upf_usage.mean()

# Plot the UPF usage
plt.figure(figsize=(20, 10))
plt.plot(upf_usage.index, upf_usage.values, marker='o', linestyle='-')
plt.axhline(y=average_upf_usage, color='r', linestyle='--', label=f'Average UPF usage: {average_upf_usage:.2f}')
plt.xlabel('Time Instance', fontsize=20)
plt.ylabel('Number of Active UPFs', fontsize=20)
plt.title('UPF Usage', fontsize=25)
plt.legend()
plt.grid(True)
plt.yticks(range(int(upf_usage.min()), int(upf_usage.max()) + 1))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)

#Save plot
plt.savefig('upf_average.png', format='png')
