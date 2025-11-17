import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


# Load the Excel file
file_path = 'D:\\Romo\\Masks\\Romo_Results_Femur.xls'
df = pd.read_excel(file_path, sheet_name=1)

# Define columns
subject_col = 'Sub'
trial_col = 'Trial'
parameters = ['Epi_iBMC', 'Epi_iBMD', 'Met_iBMC', 'Met_iBMD', 'Dia_iBMC', 'Dia_iBMD']
bmd_params = ['Epi_iBMD', 'Met_iBMD', 'Dia_iBMD']  # For median/IQR and combined

# Filter for trials 1, 4, and 5, and remove unwanted subjects
df_filtered = df[df[trial_col].isin([1, 4, 5])]
df_filtered = df_filtered[~df_filtered[subject_col].isin([102, 108])]

# Create subplots (6 + 1 for combined BMD)
fig, axes = plt.subplots(4, 2, figsize=(8, 10))
axes = axes.flatten()

# Helper for median and IQR
def describe_stat(label, data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    print(f"{label} — Median: {median:.3f}, IQR: ({q1:.3f}, {q3:.3f})")

# Plot individual parameters
for i, param in enumerate(parameters):
    ax = axes[i]
    
    trial_1 = df_filtered[df_filtered[trial_col] == 1].set_index(subject_col)[param]
    trial_4 = df_filtered[df_filtered[trial_col] == 4].set_index(subject_col)[param]
    trial_5 = df_filtered[df_filtered[trial_col] == 5].set_index(subject_col)[param]
    
    subjects = trial_1.index.intersection(trial_4.index).intersection(trial_5.index)
    trial_1 = trial_1.loc[subjects]
    trial_4 = trial_4.loc[subjects]
    trial_5 = trial_5.loc[subjects]

    x = np.array([0, 0.2, 0.4])
    width = 0.15

    ax.bar(x[0], trial_1.mean(), width, label='0', color='g', alpha=0.7)
    ax.bar(x[1], trial_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
    ax.bar(x[2], trial_5.mean(), width, label='24 ', color='r', alpha=0.7)

    for subj in subjects:
        ax.scatter(x[0], trial_1[subj], color='black', s=50)
        ax.scatter(x[1], trial_4[subj], color='black', s=50)
        ax.scatter(x[2], trial_5[subj], color='black', s=50)
        ax.plot(x, [trial_1[subj], trial_4[subj], trial_5[subj]], 'k-', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(['0', '12\nmonths', '24 '])
    ax.set_title(param)

    # Wilcoxon test
    stat, p_value = wilcoxon(trial_4, trial_5)
    print(f"{param} — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")

    # Median and IQR for BMDs
    if param in bmd_params:
        describe_stat(f"{param} @ 0", trial_1)
        describe_stat(f"{param} @ 12", trial_4)
        describe_stat(f"{param} @ 24", trial_5)

# ---- Combined BMD ----
combined = {}
for trial in [1, 4, 5]:
    trial_data = df_filtered[df_filtered[trial_col] == trial].set_index(subject_col)
    trial_data = trial_data[bmd_params]
    trial_data = trial_data.dropna()
    trial_data['CombinedBMD'] = trial_data.mean(axis=1)
    combined[trial] = trial_data['CombinedBMD']

subjects_combined = combined[1].index.intersection(combined[4].index).intersection(combined[5].index)
comb_1 = combined[1].loc[subjects_combined]
comb_4 = combined[4].loc[subjects_combined]
comb_5 = combined[5].loc[subjects_combined]

# Plot combined BMD
ax = axes[6]
x = np.array([0, 0.2, 0.4])
width = 0.15

ax.bar(x[0], comb_1.mean(), width, label='0', color='g', alpha=0.7)
ax.bar(x[1], comb_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
ax.bar(x[2], comb_5.mean(), width, label='24 ', color='r', alpha=0.7)

for subj in subjects_combined:
    ax.scatter(x[0], comb_1[subj], color='black', s=50)
    ax.scatter(x[1], comb_4[subj], color='black', s=50)
    ax.scatter(x[2], comb_5[subj], color='black', s=50)
    ax.plot(x, [comb_1[subj], comb_4[subj], comb_5[subj]], 'k-', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(['0', '12\nmonths', '24 '])
ax.set_title('Combined BMD')

# Wilcoxon + stats
stat, p_value = wilcoxon(comb_4, comb_5)
print(f"Combined BMD — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")
describe_stat("Combined BMD @ 0", comb_1)
describe_stat("Combined BMD @ 12", comb_4)
describe_stat("Combined BMD @ 24", comb_5)

# Turn off empty subplot if 8th axis unused
axes[7].axis('off')

plt.tight_layout()
# plt.savefig("D:\\Romo\\Masks\\Femur_with_combined_BMD.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#########################################################################################################
# Load the Excel file
file_path = 'D:\\Romo\\Masks\\Romo_Results_Tibia.xls'
df = pd.read_excel(file_path, sheet_name=1)

# Define subject and trial columns
subject_col = 'Sub'
trial_col = 'Trial'
parameters = ['Epi_iBMC', 'Epi_iBMD', 'Met_iBMC', 'Met_iBMD', 'Dia_iBMC', 'Dia_iBMD']
bmd_params = ['Epi_iBMD', 'Met_iBMD', 'Dia_iBMD']  # For stats and combined

# Filter for trials 1, 4, and 5; remove unwanted subjects
df_filtered = df[df[trial_col].isin([1, 4, 5])]
df_filtered = df_filtered[~df_filtered[subject_col].isin([102, 108])]

# Set up subplots (6 parameters + 1 combined BMD)
fig, axes = plt.subplots(4, 2, figsize=(8, 10))
axes = axes.flatten()

# Helper function to print stats
def describe_stat(label, data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    print(f"{label} — Median: {median:.3f}, IQR: ({q1:.3f}, {q3:.3f})")

# Plot individual BMC/BMD parameters
for i, param in enumerate(parameters):
    ax = axes[i]
    
    trial_1 = df_filtered[df_filtered[trial_col] == 1].set_index(subject_col)[param]
    trial_4 = df_filtered[df_filtered[trial_col] == 4].set_index(subject_col)[param]
    trial_5 = df_filtered[df_filtered[trial_col] == 5].set_index(subject_col)[param]
    
    subjects = trial_1.index.intersection(trial_4.index).intersection(trial_5.index)
    trial_1 = trial_1.loc[subjects]
    trial_4 = trial_4.loc[subjects]
    trial_5 = trial_5.loc[subjects]

    x = np.array([0, 0.2, 0.4])
    width = 0.15

    ax.bar(x[0], trial_1.mean(), width, label='0', color='g', alpha=0.7)
    ax.bar(x[1], trial_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
    ax.bar(x[2], trial_5.mean(), width, label='24 ', color='r', alpha=0.7)

    for subj in subjects:
        ax.scatter(x[0], trial_1[subj], color='black', s=50)
        ax.scatter(x[1], trial_4[subj], color='black', s=50)
        ax.scatter(x[2], trial_5[subj], color='black', s=50)
        ax.plot(x, [trial_1[subj], trial_4[subj], trial_5[subj]], 'k-', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(['0', '12\nmonths', '24 '])
    ax.set_title(param)

    # Wilcoxon test
    stat, p_value = wilcoxon(trial_4, trial_5)
    print(f"{param} — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")

    # Median and IQR for BMDs
    if param in bmd_params:
        describe_stat(f"{param} @ 0", trial_1)
        describe_stat(f"{param} @ 12", trial_4)
        describe_stat(f"{param} @ 24", trial_5)

# ---- Combined BMD ----
combined = {}
for trial in [1, 4, 5]:
    trial_data = df_filtered[df_filtered[trial_col] == trial].set_index(subject_col)
    trial_data = trial_data[bmd_params]
    trial_data = trial_data.dropna()
    trial_data['CombinedBMD'] = trial_data.mean(axis=1)
    combined[trial] = trial_data['CombinedBMD']

subjects_combined = combined[1].index.intersection(combined[4].index).intersection(combined[5].index)
comb_1 = combined[1].loc[subjects_combined]
comb_4 = combined[4].loc[subjects_combined]
comb_5 = combined[5].loc[subjects_combined]

# Plot combined BMD
ax = axes[6]
x = np.array([0, 0.2, 0.4])
width = 0.15

ax.bar(x[0], comb_1.mean(), width, label='0', color='g', alpha=0.7)
ax.bar(x[1], comb_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
ax.bar(x[2], comb_5.mean(), width, label='24 ', color='r', alpha=0.7)

for subj in subjects_combined:
    ax.scatter(x[0], comb_1[subj], color='black', s=50)
    ax.scatter(x[1], comb_4[subj], color='black', s=50)
    ax.scatter(x[2], comb_5[subj], color='black', s=50)
    ax.plot(x, [comb_1[subj], comb_4[subj], comb_5[subj]], 'k-', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(['0', '12\nmonths', '24 '])
ax.set_title('Combined BMD')

# Wilcoxon + stats
stat, p_value = wilcoxon(comb_4, comb_5)
print(f"Combined BMD — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")
describe_stat("Combined BMD @ 0", comb_1)
describe_stat("Combined BMD @ 12", comb_4)
describe_stat("Combined BMD @ 24", comb_5)

# Hide empty 8th subplot
axes[7].axis('off')

plt.tight_layout()
# plt.savefig("D:\\Romo\\Masks\\Tibia_with_combined_BMD.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#########################################################################################################

# Load the Excel file
file_path = 'D:\\Romo\\Masks\\Romo_Results_Hip.xls'
df = pd.read_excel(file_path, sheet_name=2)

# Define subject and trial columns
subject_col = 'Sub'
trial_col = 'Trial'

# Parameters to plot
parameters = ['FNiBMC', 'FNiBMD', 'TriBMC', 'TriBMD']
bmd_params = ['FNiBMD', 'TriBMD']

# Filter for trials 1, 4, 5 and remove unwanted subjects
df_filtered = df[df[trial_col].isin([1, 4, 5])]
df_filtered = df_filtered[~df_filtered[subject_col].isin([102, 108])]

# Create 3x2 subplots (extra subplot for combined BMD)
fig, axes = plt.subplots(3, 2, figsize=(8, 9))
axes = axes.flatten()

# Helper for stats
def describe_stat(label, data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    print(f"{label} — Median: {median:.3f}, IQR: ({q1:.3f}, {q3:.3f})")

# Individual parameters
for i, param in enumerate(parameters):
    ax = axes[i]
    
    trial_1 = df_filtered[df_filtered[trial_col] == 1].set_index(subject_col)[param]
    trial_4 = df_filtered[df_filtered[trial_col] == 4].set_index(subject_col)[param]
    trial_5 = df_filtered[df_filtered[trial_col] == 5].set_index(subject_col)[param]

    subjects = trial_1.index.intersection(trial_4.index).intersection(trial_5.index)
    trial_1 = trial_1.loc[subjects]
    trial_4 = trial_4.loc[subjects]
    trial_5 = trial_5.loc[subjects]

    x = np.array([0, 0.2, 0.4])
    width = 0.15

    ax.bar(x[0], trial_1.mean(), width, label='0', color='g', alpha=0.7)
    ax.bar(x[1], trial_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
    ax.bar(x[2], trial_5.mean(), width, label='24 ', color='r', alpha=0.7)

    for subj in subjects:
        ax.scatter(x[0], trial_1[subj], color='black', s=50)
        ax.scatter(x[1], trial_4[subj], color='black', s=50)
        ax.scatter(x[2], trial_5[subj], color='black', s=50)
        ax.plot(x, [trial_1[subj], trial_4[subj], trial_5[subj]], 'k-', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(['0', '12\nmonths', '24 '])
    ax.set_title(param)

    # Wilcoxon test
    stat, p_value = wilcoxon(trial_4, trial_5)
    print(f"{param} — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")

    # Med + IQR
    if param in bmd_params:
        describe_stat(f"{param} @ 0", trial_1)
        describe_stat(f"{param} @ 12", trial_4)
        describe_stat(f"{param} @ 24", trial_5)

# ---- Combined BMD ----
combined = {}
for trial in [1, 4, 5]:
    trial_data = df_filtered[df_filtered[trial_col] == trial].set_index(subject_col)
    trial_data = trial_data[bmd_params]
    trial_data = trial_data.dropna()
    trial_data['CombinedBMD'] = trial_data.mean(axis=1)
    combined[trial] = trial_data['CombinedBMD']

subjects_combined = combined[1].index.intersection(combined[4].index).intersection(combined[5].index)
comb_1 = combined[1].loc[subjects_combined]
comb_4 = combined[4].loc[subjects_combined]
comb_5 = combined[5].loc[subjects_combined]

# Plot combined BMD
ax = axes[4]
x = np.array([0, 0.2, 0.4])
width = 0.15

ax.bar(x[0], comb_1.mean(), width, label='0', color='g', alpha=0.7)
ax.bar(x[1], comb_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
ax.bar(x[2], comb_5.mean(), width, label='24 ', color='r', alpha=0.7)

for subj in subjects_combined:
    ax.scatter(x[0], comb_1[subj], color='black', s=50)
    ax.scatter(x[1], comb_4[subj], color='black', s=50)
    ax.scatter(x[2], comb_5[subj], color='black', s=50)
    ax.plot(x, [comb_1[subj], comb_4[subj], comb_5[subj]], 'k-', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(['0', '12\nmonths', '24 '])
ax.set_title("Combined BMD")

# Stats
stat, p_value = wilcoxon(comb_4, comb_5)
print(f"Combined BMD — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")
describe_stat("Combined BMD @ 0", comb_1)
describe_stat("Combined BMD @ 12", comb_4)
describe_stat("Combined BMD @ 24", comb_5)

# Disable unused last subplot
axes[5].axis('off')

plt.tight_layout()
# plt.savefig("D:\\Romo\\Masks\\Hip_with_combined_BMD.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


#########################################################################################################
# Load the Excel file
file_path = 'D:\\Romo\\Masks\\Romo Tibia FEA Results.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Define subject and Trial columns
subject_col = 'Sub'
trial_col = 'Trial'

# Parameters and titles
parameters = ['UltT', 'K']
titles = ["Torsional strength (Nm)", "Stiffness (°)"]

# Filter data for trials 1, 4, and 5
df_filtered = df[df[trial_col].isin([1, 4, 5])]
df_filtered = df_filtered[~df_filtered[subject_col].isin([102, 108])]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes = axes.flatten()

for i, param in enumerate(parameters):
    ax = axes[i]

    # Extract values for each trial
    trial_1 = df_filtered[df_filtered[trial_col] == 1].set_index(subject_col)[param]
    trial_4 = df_filtered[df_filtered[trial_col] == 4].set_index(subject_col)[param]
    trial_5 = df_filtered[df_filtered[trial_col] == 5].set_index(subject_col)[param]

    # Keep only subjects present in all three trials
    subjects = trial_1.index.intersection(trial_4.index).intersection(trial_5.index)
    trial_1 = trial_1.loc[subjects]
    trial_4 = trial_4.loc[subjects]
    trial_5 = trial_5.loc[subjects]

    # --- Print Median, Q1, and Q3 ---
    print(f"\n--- Statistics for {titles[i]} ---")

    # Trial 1
    median_1 = trial_1.median()
    Q1_1 = trial_1.quantile(0.25)
    Q3_1 = trial_1.quantile(0.75)
    print(f"Trial 1 (0 months): Median = {median_1:.2f}, Q1 = {Q1_1:.2f}, Q3 = {Q3_1:.2f}")

    # Trial 4
    median_4 = trial_4.median()
    Q1_4 = trial_4.quantile(0.25)
    Q3_4 = trial_4.quantile(0.75)
    print(f"Trial 4 (12 months): Median = {median_4:.2f}, Q1 = {Q1_4:.2f}, Q3 = {Q3_4:.2f}")

    # Trial 5
    median_5 = trial_5.median()
    Q1_5 = trial_5.quantile(0.25)
    Q3_5 = trial_5.quantile(0.75)
    print(f"Trial 5 (24 months): Median = {median_5:.2f}, Q1 = {Q1_5:.2f}, Q3 = {Q3_5:.2f}")

    # Bar plot positions
    x = np.array([0, 0.2, 0.4])
    width = 0.15

    # Bar plots for mean values
    ax.bar(x[0], trial_1.mean(), width, label='0', color='g', alpha=0.7)
    ax.bar(x[1], trial_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
    ax.bar(x[2], trial_5.mean(), width, label='24 ', color='r', alpha=0.7)

    # Scatter and connecting lines
    for subj in subjects:
        ax.scatter(x[0], trial_1[subj], color='black', s=50)
        ax.scatter(x[1], trial_4[subj], color='black', s=50)
        ax.scatter(x[2], trial_5[subj], color='black', s=50)
        ax.plot(x, [trial_1[subj], trial_4[subj], trial_5[subj]], 'k-', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(['0', '12\nmonths', '24 '])
    ax.set_title(titles[i])

    # ---- Wilcoxon Signed-Rank Test ----
    stat, p_value = wilcoxon(trial_4, trial_5)
    print(f"Wilcoxon test (12 vs 24 months) for {titles[i]}: stat = {stat:.3f}, p = {p_value:.4f}")

plt.tight_layout()
plt.savefig("D:\\Romo\\Masks\\Tibia_FEA.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#########################################################################################################
# Load the Excel file (placeholder - update with your actual file path)
file_path = 'D:\\Romo\\Masks\\Hip FEA Results.xlsx' 
df = pd.read_excel(file_path, sheet_name=0)

# Define subject and Trial columns
subject_col = 'Sub'
trial_col = 'Trial'

# Parameters and titles
parameters = ['FailureLoad', '% Change']
titles = ["Strength (N)", "% Change from baseline (N)"]

# Filter for trials 1, 4, and 5, and exclude subjects
df_filtered = df[df[trial_col].isin([1, 4, 5])]
df_filtered = df_filtered[~df_filtered[subject_col].isin([102, 108])]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes = axes.flatten()

for i, param in enumerate(parameters):
    ax = axes[i]

    # Extract and align subject data
    trial_1 = df_filtered[df_filtered[trial_col] == 1].set_index(subject_col)[param]
    trial_4 = df_filtered[df_filtered[trial_col] == 4].set_index(subject_col)[param]
    trial_5 = df_filtered[df_filtered[trial_col] == 5].set_index(subject_col)[param]

    subjects = trial_1.index.intersection(trial_4.index).intersection(trial_5.index)
    trial_1 = trial_1.loc[subjects]
    trial_4 = trial_4.loc[subjects]
    trial_5 = trial_5.loc[subjects]

    # Bar positions and means
    x = np.array([0, 0.2, 0.4])
    width = 0.15

    ax.bar(x[0], trial_1.mean(), width, label='0', color='g', alpha=0.7)
    ax.bar(x[1], trial_4.mean(), width, label='12\nmonths', color='b', alpha=0.7)
    ax.bar(x[2], trial_5.mean(), width, label='24 ', color='r', alpha=0.7)

    # Scatter and lines per subject
    for subj in subjects:
        ax.scatter(x[0], trial_1[subj], color='black', s=50)
        ax.scatter(x[1], trial_4[subj], color='black', s=50)
        ax.scatter(x[2], trial_5[subj], color='black', s=50)
        ax.plot(x, [trial_1[subj], trial_4[subj], trial_5[subj]], 'k-', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(['0', '12\nmonths', '24 '])
    ax.set_title(titles[i])

    # --- Print Medians and IQRs ---
    print(f"\n--- Statistics for {titles[i]} ---")

    # Trial 1
    median_1 = trial_1.median()
    Q1_1 = trial_1.quantile(0.25)
    Q3_1 = trial_1.quantile(0.75)
    iqr_1 = Q3_1 - Q1_1
    print(f"Trial 1 (0 months): Median = {median_1:.2f}, IQR = [{Q1_1:.2f}, {Q3_1:.2f}]")

    # Trial 4
    median_4 = trial_4.median()
    Q1_4 = trial_4.quantile(0.25)
    Q3_4 = trial_4.quantile(0.75)
    iqr_4 = Q3_4 - Q1_4
    print(f"Trial 4 (12 months): Median = {median_4:.2f}, IQR = [{Q1_4:.2f}, {Q3_4:.2f}]")

    # Trial 5
    median_5 = trial_5.median()
    Q1_5 = trial_5.quantile(0.25)
    Q3_5 = trial_5.quantile(0.75)
    iqr_5 = Q3_5 - Q1_5
    print(f"Trial 5 (24 months): Median = {median_5:.2f}, IQR = [{Q1_5:.2f}, {Q3_5:.2f}]")

    # ---- Wilcoxon Signed-Rank Test (12 vs 24 months) ----
    stat, p_value = wilcoxon(trial_4, trial_5)
    print(f"Wilcoxon test (12 vs 24 months) for {titles[i]}: stat = {stat:.3f}, p = {p_value:.4f}")

plt.tight_layout()
plt.savefig("D:\\Romo\\Masks\\Hip_FEA.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

#########################################################################################################
# Load the Excel file
file_path = 'D:\\Romo\\Masks\\RomoInSCI-DXADataReport_DATA_LABELS_2024-08-15_1538.xlsx'
df = pd.read_excel(file_path, sheet_name=1)

# Define subject and trial columns
subject_col = 'ID'
trial_col = 'Visit'

# Parameters and timepoints
parameters = ['Spine BMD', 'Primary Hip: Total BMD', 'Primary Hip: FN BMD']
timepoints = ['M00', 'M12', 'M24']

# Filter for the desired timepoints
df_filtered = df[df[trial_col].isin(timepoints)]

# Remove specific subjects if needed
df_filtered = df_filtered[~df_filtered[subject_col].isin([102, 108])]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes = axes.flatten()

for i, param in enumerate(parameters):
    ax = axes[i]
    
    # Extract values
    trial_0 = df_filtered[df_filtered[trial_col] == 'M00'].set_index(subject_col)[param]
    trial_12 = df_filtered[df_filtered[trial_col] == 'M12'].set_index(subject_col)[param]
    trial_24 = df_filtered[df_filtered[trial_col] == 'M24'].set_index(subject_col)[param]
    
    # Keep only subjects with all three timepoints
    subjects = trial_0.index.intersection(trial_12.index).intersection(trial_24.index)
    trial_0 = trial_0.loc[subjects]
    trial_12 = trial_12.loc[subjects]
    trial_24 = trial_24.loc[subjects]

    # Plotting
    x = np.array([0, 0.2, 0.4])
    width = 0.15
    ax.bar(x[0], trial_0.mean(), width, label='0', color='g', alpha=0.7)
    ax.bar(x[1], trial_12.mean(), width, label='12\nmonths', color='b', alpha=0.7)
    ax.bar(x[2], trial_24.mean(), width, label='24 ', color='r', alpha=0.7)

    for subj in subjects:
        ax.scatter(x[0], trial_0[subj], color='black', s=50)
        ax.scatter(x[1], trial_12[subj], color='black', s=50)
        ax.scatter(x[2], trial_24[subj], color='black', s=50)
        ax.plot(x, [trial_0[subj], trial_12[subj], trial_24[subj]], 'k-', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(['0', '12\nmonths', '24 '])
    ax.set_title(param)

    # Wilcoxon test
    stat, p_value = wilcoxon(trial_12, trial_24)
    print(f"{param} — Wilcoxon test (12 vs 24 months): stat = {stat:.3f}, p = {p_value:.4f}")

    # Median and IQR
    for label, values in zip(['0', '12', '24'], [trial_0, trial_12, trial_24]):
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        print(f"{param} @ {label} — Median: {median:.3f}, IQR: ({q1:.3f}, {q3:.3f})")

plt.tight_layout()
plt.savefig("D:\\Romo\\Masks\\DXA_BMD_Chart.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
#########################################################################################################

#########################################################################################################

#########################################################################################################
