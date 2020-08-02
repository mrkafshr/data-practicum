# Mark Fisher
# Stockton University - Data Practicum
# Visualization of data in enterprise reporting systems

print("\n")
print("### Mark Fisher")
print("### Stockton University - Data Practicum")
print("### Summer 2020")
print("### Visualization of data in enterprise reporting systems")

# Load required libraries
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Color setup for chart
chart_colors = ['#CFC8A3', '#ED7272']
sns.set_palette(chart_colors)
sns.palplot(sns.color_palette())

# Create function that will put bar values as text on the chart
# Credit: https://stackoverflow.com/a/56780852
def show_values_on_bars(axs, space=0.4, fontsize=10, heightValue=0.5):
    def _show_on_single_plot(ax):
        for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + heightValue
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center", size=fontsize, weight='bold')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

print("\n")
print("### Loading dataset...")

# Load report migration dataset
df = pd.read_csv('data/report_migration_data.csv')
print("### Dataset loaded:")
print(df.info())

print("\n")
print("### Cleaning dataset...")

# Drop every row that doesn't have an 'ARGOS_PATH' or 'DISCOVERER_EUL' value.
df = df[pd.notnull(df['ARGOS_PATH'])]
df = df[pd.notnull(df['DISCOVERER_EUL'])]

# Exclude unneeded EULs from dataset.
# AUTOMIC is another report/data processing system, and
# FINANCE and HISTORY are EULs that were not used at all in Discoverer
bad_euls = ['AUTOMIC', 'FINANCE', 'HISTORY']
df = df[~df[['DISCOVERER_EUL']].isin(bad_euls).all(axis=1)]

# Exclude unneeded Discoverer worksheets from dataset.
# These reports were limited to specific student ID #'s,
# and were no longer being used.
unneeded_worksheets = ['Hard-coded to specific Z']
df = df[~df[['DISCOVERER_WORKSHEET']].isin(unneeded_worksheets).all(axis=1)]

# Exclude unneeded Argos data from dataset.
# "FGBTRND/FPBPOHD" are form(s) in Ellucian Banner that accomplish
# the same thing the Discoverer report did, so no Argos report is needed.
# Other paths are self explanatory (Not Needed, Use Banner), either not needed
# at all, or an equivalent Banner form/report is available.
unneeded_paths = ['FGBTRND/FPBPOHD', 'Not Needed.', 'Use Banner', 'Not Needed']
df = df[~df[['ARGOS_PATH']].isin(unneeded_paths).all(axis=1)]

# Removing custom/old/unneeded data, as marked by "ARGOS_DATABLOCK" column.
unneeded_datablocks = ['Still private in Discoverer', 'Uses old STOCKTON.Q_STUDY table', 'Not Needed', 'Untouched since 2013']
df = df[~df[['ARGOS_DATABLOCK']].isin(unneeded_datablocks).all(axis=1)]

# Removing old/unneeded data, as marked by "ARGOS_REPORT_NAME" column.
unneeded_reports = ['Broken in Discoverer', 'Not needed.']
df = df[~df[['ARGOS_REPORT_NAME']].isin(unneeded_reports).all(axis=1)]

# Remove any unmigrated reports from dataset
unfinished_reports = ['Y']
df = df[~df[['INPROG_IND']].isin(unfinished_reports).all(axis=1)]

# Reconfigure dataset to be
#   SYSTEM_FLAG: Discoverer or Argos
#   EUL: DISCOVERER_EUL
#   BLOCK: DISCOVERER_WORKBOOK/ARGOS_DATABLOCK
#   REPORT: DISCOVERER_WORKSHEET/ARGOS_REPORT_NAME
#   PATH: ARGOS_PATH
discoverer_reports = df[['DISCOVERER_EUL', 'DISCOVERER_WORKBOOK', 'DISCOVERER_WORKSHEET', 'ARGOS_PATH', 'INPROG_IND']]
discoverer_reports = discoverer_reports[['DISCOVERER_EUL', 'DISCOVERER_WORKBOOK', 'DISCOVERER_WORKSHEET', 'ARGOS_PATH']]
discoverer_reports.rename(columns = {'DISCOVERER_EUL':          'EUL',
                                     'DISCOVERER_WORKBOOK':     'BLOCK',
                                     'DISCOVERER_WORKSHEET':    'REPORT',
                                     'ARGOS_PATH':              'ARGOS_PATH'}, inplace = True)
discoverer_reports['SYSTEM_FLAG'] = 'Discoverer'

argos_reports = df[['DISCOVERER_EUL', 'DISCOVERER_WORKBOOK', 'DISCOVERER_WORKSHEET', 'ARGOS_PATH', 'ARGOS_DATABLOCK', 'ARGOS_REPORT_NAME', 'INPROG_IND']]
argos_reports = argos_reports[['DISCOVERER_EUL', 'ARGOS_DATABLOCK', 'ARGOS_REPORT_NAME', 'ARGOS_PATH']]
argos_reports.rename(columns = {'DISCOVERER_EUL':       'EUL',
                                'ARGOS_DATABLOCK':      'BLOCK',
                                'ARGOS_REPORT_NAME':    'REPORT',
                                'ARGOS_PATH':           'ARGOS_PATH'}, inplace = True)
argos_reports['SYSTEM_FLAG'] = 'Argos'

# UH OH! Issue with Discoverer - "Sheet 1" defualt value problem
# Update 'REPORT' column
discoverer_reports['REPORT'] = discoverer_reports['BLOCK'].str.cat(discoverer_reports['REPORT'],sep=";",na_rep="Sheet1")
argos_reports['REPORT'] = argos_reports['BLOCK'].str.cat(argos_reports['REPORT'],sep=";",na_rep="Dashboard")

# Join datasets together & cleanup duplicate reports before counting
reports = [discoverer_reports, argos_reports]
reports_by_eul = pd.concat(reports)
reports_by_eul.drop_duplicates(subset=['SYSTEM_FLAG', 'EUL', 'REPORT'], inplace=True, keep='first')
reports_by_eul.reset_index(inplace=True)

# Get counts needed for charting
reports_by_eul['Count_by_EUL'] = reports_by_eul.groupby(['SYSTEM_FLAG', 'EUL']).REPORT.transform('nunique')
reports_by_eul['Count_by_Path'] = reports_by_eul.groupby(['SYSTEM_FLAG', 'ARGOS_PATH']).REPORT.transform('nunique')
reports_by_eul['Count_by_EUL_Path'] = reports_by_eul.groupby(['SYSTEM_FLAG', 'ARGOS_PATH', 'EUL']).REPORT.transform('nunique')
reports_by_eul['Total_by_System'] = reports_by_eul.groupby(['SYSTEM_FLAG']).REPORT.transform('nunique')

# Sort & save cleaned dataset with counts
reports_by_eul = reports_by_eul.sort_values(["EUL", "ARGOS_PATH", "SYSTEM_FLAG"])
reports_by_eul.to_csv("data/sanitized_dataset.csv")

print('Saving dataset in "data/sanitized_dataset.csv"\n')
print("### Finished cleaning and organizing dataset.")
print("### Creating visualizations...")
#############                                             #############
#########      Visualizations using Seaborn & Matplotlib      #########
#############                                             #############

### Visualization 1
# Create chart
count_by_system = sns.catplot(x='SYSTEM_FLAG',
                y="Total_by_System",
                #hue='SYSTEM_FLAG',
                hue_order= ['Discoverer', 'Argos'],
                order = ['Discoverer', 'Argos'],
                data=reports_by_eul,
                kind="bar",
                ci=None,
                legend_out=False)

# Chart customization
show_values_on_bars(count_by_system.ax, fontsize=30, heightValue=1)
plt.title('Reports in Discoverer vs. Argos', fontdict={'fontsize':30, 'weight':'bold'})
plt.text(-0.45, 1900, 'Reporting System',
         fontsize=20)
plt.legend(['Discoverer', 'Argos'], loc=2, fontsize=20)

plt.xlabel("Reporting System",size=25,weight='bold')
plt.xticks(size = 18, weight = 'bold')

plt.ylabel("# of Unique Reports",size=25,weight='bold')
plt.yticks(size = 20, weight = 'bold')
# plt.ylim((0, 550))

plt.gcf().set_size_inches(35, 52.5, forward=True)
plt.savefig('visualizations/count_by_system.png', dpi=250, bbox_inches='tight')
print('Saving visualization in "visualizations/count_by_system.png"')

# Clear plot
plt.clf()

### Visualization 2
###
# Create chart
count_by_eul = sns.catplot(x="EUL",
                y="Count_by_EUL",
                hue='SYSTEM_FLAG',
                hue_order= ['Discoverer', 'Argos'],
                data=reports_by_eul,
                kind="bar",
                ci=None,
                legend_out=False)

# Chart customization
show_values_on_bars(count_by_eul.ax, fontsize=30, heightValue=1)
plt.title('Reports in Discoverer vs. Argos, by Discoverer folder structure', fontdict={'fontsize':30, 'weight':'bold'})
plt.text(-0.45, 550, 'Reporting System',
         fontsize=20)
plt.legend(loc=2, fontsize=20)

plt.xlabel("End User Layer (EUL)",size=25,weight='bold')
plt.xticks(size = 18, weight = 'bold')

plt.ylabel("# of Unique Reports",size=25,weight='bold')
plt.yticks(size = 20, weight = 'bold')
plt.ylim((0, 550))

plt.gcf().set_size_inches(50, 22.5, forward=True)
plt.savefig('visualizations/count_by_system_and_eul.png', dpi=250, bbox_inches='tight')
print('Saving visualization in "visualizations/count_by_system_and_eul.png"')

# Clear plot
plt.clf()

### Visualization 3
###
# sort dataset by Argos PATH
reports_by_eul = reports_by_eul.sort_values(["ARGOS_PATH", "SYSTEM_FLAG"])
# Create chart
count_by_path = sns.catplot(x="ARGOS_PATH",
                y="Count_by_Path",
                hue='SYSTEM_FLAG',
                hue_order= ['Discoverer', 'Argos'],
                data=reports_by_eul,
                kind="bar",
                ci=None,
                legend_out=False)

# Chart customization
show_values_on_bars(count_by_path.ax, fontsize=22, heightValue=1)
count_by_path.set_xticklabels(rotation=90)
plt.title('Reports in Discoverer vs. Argos, by Argos folder structure', fontdict={'fontsize':30, 'weight':'bold'})
plt.text(-0.25, 250, 'Reporting System',
         fontsize=20)
plt.legend(loc=2, fontsize=20)

plt.xlabel("Argos Path",size=25,weight='bold')
plt.xticks(size = 18, weight = 'bold')

plt.ylabel("# of Unique Reports",size=25,weight='bold')
plt.yticks(size = 20, weight = 'bold')
plt.ylim((0, 250))

plt.gcf().set_size_inches(120, 22.5, forward=True)
plt.savefig('visualizations/count_by_system_and_path.png', dpi=300, bbox_inches='tight')
print('Saving visualization in "visualizations/count_by_system_and_path.png"')

# Clear plot
plt.clf()

### Visualization 4
# Sort data by eul, then Argos paths
reports_by_eul = reports_by_eul.sort_values(["EUL", "ARGOS_PATH", "SYSTEM_FLAG"])
# Create chart
g = sns.FacetGrid(reports_by_eul, col='EUL', col_wrap=3, height=15, sharex=False, sharey=False, legend_out=False)
g.map(sns.barplot, 'ARGOS_PATH', 'Count_by_EUL_Path', 'SYSTEM_FLAG', palette=chart_colors, hue_order= ['Discoverer', 'Argos'])

# Chart customization
g.fig.suptitle('Reports in Discoverer vs. Argos, \n by Discoverer and Argos folder structure', fontsize=48)
g.set_titles('End User Layer = {col_name}')

axes = g.axes.flatten()
axes[6].set_xlabel("Argos Path", size=25, weight = 'bold')
axes[7].set_xlabel("Argos Path", size=25, weight = 'bold')
axes[8].set_xlabel("Argos Path", size=25, weight = 'bold')

axes[0].set_ylabel("# of Unique Reports", size=25, weight = 'bold')
axes[3].set_ylabel("# of Unique Reports", size=25, weight = 'bold')
axes[6].set_ylabel("# of Unique Reports", size=25, weight = 'bold')

axes[0].legend()
axes[3].legend()
axes[6].legend()

for ax in g.axes.ravel():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=17, weight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
for ax in g.axes.ravel():
    show_values_on_bars(ax, fontsize=21, heightValue = 0.25)

plt.savefig('visualizations/count_by_system_and_eul_and_path.png', dpi=80)
print('Saving visualization in "visualizations/count_by_system_and_eul_and_path.png"\n')

print("### Finished. Exiting...")
