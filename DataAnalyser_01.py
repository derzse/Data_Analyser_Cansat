import os 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager as fm, rcParams, rc
from matplotlib.lines import Line2D
import seaborn as sns
import time
import datetime
from cycler import cycler
from seaborn import colors
import datetime
import matplotlib.patches as mpatches


# parameters
matplotlib.rcParams['font.family'] = "roboto"
matplotlib.rcParams['pdf.fonttype'] = 42 # Set for TrueType Fonts for pdf
matplotlib.rcParams['ps.fonttype'] = 42 # Set for TrueType Fonts for ps
fpath = os.path.join(rcParams["datapath"],  
                     "fonts/ttf/Roboto-Regular.ttf")
prop = fm.FontProperties(fname = fpath) 

sns.set(style = "darkgrid")
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 8

sns.set_context({"axes.linewidth": 0.75,
                 "xtick.major.size": 4, 
                 "ytick.major.size": 8})
pallete1 = sns.color_palette("RdPu_r", n_colors = 4)
pallete2 = sns.color_palette("YlOrRd_r", n_colors = 3)
pallete4 = sns.color_palette("GnBu_r", n_colors = 3)
pallete3 = sns.color_palette("YlOrBr", n_colors = 3)

gsm_packeges = np.array([38, 53, 68, 97, 111, 126, 141, 156, 171, 186, 201, 216, 232, 262, 291, 305, 320, 336, 351, 367, 382, 398, 428, 442, 458, 487, 708, 765, 807, 882, 1041, 1055, 1070, 1161, 1191, 1251,  1265])

#
# dataframe loading and Time of Flight calculations
#
df = pd.read_csv('data_cansat_clean.csv', sep=';')
launchtime = datetime.datetime(2021, 5, 14, 16, 5, 0).timestamp()
firstRuntime = df['Runtime'].iloc[0]
df['TimeOfFlight'] = launchtime + df['Runtime'] - firstRuntime
df['bmp388Pres'] = 100*df['bmp388Pres']
#print(df)
for i in range(df.shape[0]):
    if df['PackID'].iloc[i] in gsm_packeges:
        df.at[i,'GSM'] = True
    else:
        df.at[i,'GSM'] = False
df.index = pd.to_datetime(df['TimeOfFlight'], unit = 's')

print("Full data set (cleaned):")
print("  + Dataframe shape: Rows {}, Columns {}".format(df.shape[0], df.shape[1]))
dt = (df.index[-1] - df.index[0])
print("  + Length of the time series: {}".format(dt))
#
# GSM data 
#
dfgsm = pd.read_csv('data_simcard.csv', sep=';')
firstRuntime = dfgsm['Runtime'].iloc[0]
dfgsm['TimeOfFlight'] = launchtime + dfgsm['Runtime'] - firstRuntime
dfgsm['bmp388Pres'] = 100*dfgsm['bmp388Pres']
#print(dfgsm)
dfgsm.index = pd.to_datetime(dfgsm['TimeOfFlight'], unit = 's')
print("GSM Data set:")
print("  + Dataframe shape (GSM): Rows {}, Columns {}".format(dfgsm.shape[0], dfgsm.shape[1]))
dtgsm = (dfgsm.index[-1] - dfgsm.index[0])
print("  + Length of the GSM time series: {}".format(dtgsm))

#
# dataframe for flight time
#
df_flight = df.iloc[389:430,:]
print("Ascend and descend timeframe:")
#print(df_flight)

print("  + Dataframe shape (ascend and descend): Rows {}, Columns {}".format(df_flight.shape[0], df_flight.shape[1]))
dt_flight = (df_flight.index[-1] - df_flight.index[0])
print("  + Length of the ascend and descend time series: {}".format(dt_flight))

dfgsm_flight = dfgsm.iloc[24:26,:]
print("Ascend and descend timeframe:")
print(dfgsm_flight)


# flight window
x0 = 1838.72 # package 391
x1 = 2037.61 # package 431

#plotting the two lines for the flight time window
p1 = plt.axvline(x=x0,color='#EF9A9A')
p2 = plt.axvline(x=x1,color='#EF9A9A')


# concatenated = pd.concat([df.assign(dataset='df'), dfgsm.assign(dataset='dfgsm')])

#
# Temperatures graphs
#

fig1, ax1 = plt.subplots(1, 1,figsize=(16, 9))
ax1.set_prop_cycle(cycler(color=pallete1))

ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
sns.despine(ax=ax1, top = True, offset = 40, trim = True)
sns.scatterplot(ax = ax1, x = 'Runtime', y = 'mcp9808Temp', data = dfgsm, marker = 'o', color="red", size=40) #, fillstyle='none') #, fillstyle='none', markeredgewidth=1.1, markeredgecolor='red')
sns.lineplot(ax = ax1, x = 'Runtime', y = 'mcp9808Temp',  data = df)#, marker = 'o', markersize=8, fillstyle='none', markeredgewidth=1.1, markeredgecolor='red', 
    # markevery=[40, 55, 70, 99, 113, 128, 143, 158, 173, 188, 203, 218, 234, 264, 293, 307, 322, 338, 353, 369, 384, 400, 430, 444, 460, 489, 710, 767, 809, 884, 1043, 1057, 1072, 1163, 1193, 1253, 1267]) # markers=True, markevery=gsm_packeges, markersize=10)
ax1.set_title('CanSat Temperature Sensor Readings (MCP9808)', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_ylabel('Temperature [°C]', fontsize = 14)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)


xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
plt.legend(labels = ["Temperature"])















#
# Temperature graphs
#
dfAlt = pd.melt(df, id_vars = 'Runtime', value_vars = ["bmp388Temp","bme280Temp","mcp9808Temp"])
dfAltGSM = pd.melt(dfgsm, id_vars = 'Runtime', value_vars = ["bmp388Temp","bme280Temp","mcp9808Temp"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete1)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 

xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Temperature Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Temperature [°C]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = ["BMP388 (I2C_1)", "BME280 (I2C_0)", "MCP9808 (I2C_1)", "SMS Readings"])








# Pressure graphs
#
dfAlt = pd.melt(df, id_vars = 'Runtime', value_vars = ["bmp388Pres","bme280Pres"])
dfAltGSM = pd.melt(dfgsm, id_vars = 'Runtime', value_vars = ["bmp388Pres","bme280Pres"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete2)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 

xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Pressure Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Pressure [hPa]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = ["BMP388 (I2C_1)", "BME280 (I2C_0)", "SMS Readings"])






# Altitude graphs
#
dfAlt = pd.melt(df, id_vars = 'Runtime', value_vars = ["lps25mBar","bmp388Alt","bme280Alt"])
dfAltGSM = pd.melt(dfgsm, id_vars = 'Runtime', value_vars = ["lps25mBar","bmp388Alt","bme280Alt"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete3)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 

xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Altitude Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Altitude [m]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = ["LPS25 (I2C_0)", "BMP388 (I2C_1)", "BME280 (I2C_0)", "SMS Readings"])










































#
# Temperatures graphs
#

fig1, ax1 = plt.subplots(1, 1,figsize=(16, 9))
ax1.set_prop_cycle(cycler(color=pallete1))

ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
sns.despine(ax=ax1, top = True, offset = 40, trim = True)
sns.scatterplot(ax = ax1, x = 'Runtime', y = 'mcp9808Temp', data = dfgsm, marker = 'o', color="red", size=40) #, fillstyle='none') #, fillstyle='none', markeredgewidth=1.1, markeredgecolor='red')
sns.lineplot(ax = ax1, x = 'Runtime', y = 'mcp9808Temp',  data = df)#, marker = 'o', markersize=8, fillstyle='none', markeredgewidth=1.1, markeredgecolor='red', 
    # markevery=[40, 55, 70, 99, 113, 128, 143, 158, 173, 188, 203, 218, 234, 264, 293, 307, 322, 338, 353, 369, 384, 400, 430, 444, 460, 489, 710, 767, 809, 884, 1043, 1057, 1072, 1163, 1193, 1253, 1267]) # markers=True, markevery=gsm_packeges, markersize=10)
ax1.set_title('CanSat Temperature Sensor Readings (MCP9808)', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_ylabel('Temperature [°C]', fontsize = 14)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)


xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
plt.legend(labels = ["Temperature"])















#
# Temperature graphs
#
dfAlt = pd.melt(df_flight, id_vars = 'Runtime', value_vars = ["bmp388Temp","bme280Temp","mcp9808Temp"])
dfAltGSM = pd.melt(dfgsm_flight, id_vars = 'Runtime', value_vars = ["bmp388Temp","bme280Temp","mcp9808Temp"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete1)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 


ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Temperature Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Temperature [°C]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = ["BMP388 (I2C_1)", "BME280 (I2C_0)", "MCP9808 (I2C_1)", "SMS Readings"])








# Pressure graphs
#
dfAlt = pd.melt(df_flight, id_vars = 'Runtime', value_vars = ["bmp388Pres","bme280Pres"])
dfAltGSM = pd.melt(dfgsm_flight, id_vars = 'Runtime', value_vars = ["bmp388Pres","bme280Pres"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete2)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 


ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Pressure Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Pressure [hPa]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = ["BMP388 (I2C_1)", "BME280 (I2C_0)", "SMS Readings"])






# Altitude graphs
#
dfAlt = pd.melt(df_flight, id_vars = 'Runtime', value_vars = ["lps25mBar","bmp388Alt","bme280Alt"])
dfAltGSM = pd.melt(dfgsm_flight, id_vars = 'Runtime', value_vars = ["lps25mBar","bmp388Alt","bme280Alt"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete3)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 


ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Altitude Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Altitude [m]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = ["LPS25 (I2C_0)", "BMP388 (I2C_1)", "BME280 (I2C_0)", "SMS Readings"])




# Altitude graphs
#
dfAlt = pd.melt(df_flight, id_vars = 'Runtime', value_vars = ["bmp388Alt"])
dfAltGSM = pd.melt(dfgsm_flight, id_vars = 'Runtime', value_vars = ["bmp388Alt"])
#dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
print(dfAlt)

plt.figure(figsize = (15,9))
sns.set_palette(pallete3)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'value', hue = 'variable', data = dfAlt)
ax1 = sns.scatterplot(ax = ax1, x = 'Runtime', y = 'value', data = dfAltGSM, marker = 'o', color="black", s=25) 


ax2 = ax1.twiny()
ax2.set_xlabel('Local time [H:M:S]', fontsize = 13)
ax2.set_xticks( ax1.get_xticks() )
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels([datetime.datetime.utcfromtimestamp(x + launchtime).strftime("%H:%M:%S") for x in ax1.get_xticks()])
ax2.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
ax2.grid(False)
ax1.set_title('CanSat 2021 Altitude Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18,
                       pad=20)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Altitude [m]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 1, 
                labelsize = 10)
handles, labels = ax1.get_legend_handles_labels()
# patch  = mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor="black", edgecolor="none")
patch = Line2D([0], [0], marker='o', color='none', label='SMS Readings',
                        markerfacecolor='black', markersize=3),
handles = handles[1:]
handles.append(patch)
ax1.legend(title = 'Sensors', handles = handles, labels = [ "BMP388", "SMS Readings"])



































#
#Barometric pressure graphs
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete2)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'bmp388Pres', data = df)
ax1.set_title('CanSat Barometric Pressure Sensor Readings (BME388)', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Pressure [hPa]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
plt.legend(labels = ["Pressure"])


#
# Temperatures graphs
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete1)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'bmp388Temp', data = df)
ax1.set_title('CanSat Temperature Sensor Readings (BME388)', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Temperature [°C]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
plt.legend(labels = ["Temperature"])

#
# Humidity graph
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete3)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'bme280Hum', data = df)
ax1.set_title('CanSat Humidity Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Humidity [%]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
plt.legend(title = 'Sensors', labels = ["Humidity"])



#
# TAltitude graphs
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete3)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'Runtime', y = 'bmp388Alt', data = df)
ax1.set_title('CanSat 2021 Altitude Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Altitude [m]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
plt.legend(labels = ["Altitude"])

# only flight and fall
#
# Temperatures graphs
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete1)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'TimeOfFlight', y = 'bmp388Alt', data = df_flight)
ax1.set_title('CanSat 2021 Altitude Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Altitude [m]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
xlim = ax1.get_xlim()
# ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
ax1.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
#reset xlim
ax1.set_xlim(xlim)
plt.legend(labels = ["Altitude"])

#
#Barometric pressure graphs
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete2)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'TimeOfFlight', y = 'bmp388Pres', data = df_flight)
ax1.set_title('CanSat Barometric Pressure Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Pressure [hPa]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
plt.legend(labels = ["Pressure"])


#
# Humidity graph
#
plt.figure(figsize = (16,9))
sns.set_palette(pallete3)
sns.despine(top = True, offset = 40, trim = True)
ax1 = sns.lineplot(x = 'TimeOfFlight', y = 'bme280Hum', data = df_flight)
ax1.set_title('CanSat Humidity Sensor Readings', 
                       fontproperties = prop,
                       fontsize = 18)
ax1.set_xlabel('Time of Flight [s]', fontsize = 14)
ax1.set_ylabel('Humidity [%]', fontsize = 14)
ax1.tick_params(which = 'major',
                direction ='out', 
                length = 8, 
                width = 2, 
                labelsize = 11)
plt.legend(title = 'Sensors', labels = ["Humidity"])







# #
# # Altitude graphs
# #
# dfAlt = pd.melt(df, id_vars = 'TimeOfFlight', value_vars = ["lps25Alt", "bme280Alt", "bmp388Alt"])
# dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')

# plt.figure(figsize = (16,9))
# sns.set_palette(pallete3)
# sns.despine(top = True, offset = 40, trim = True)
# ax1 = sns.lineplot(x = 'TimeOfFlight', y = 'value', hue = 'variable', data = dfAlt)
# ax1.set_title('CanSat Altitude Sensor Readings', 
#                        fontproperties = prop,
#                        fontsize = 18)
# ax1.set_xlabel('Time of Flight', fontsize = 14)
# ax1.set_ylabel('Altitude [m]', fontsize = 14)
# ax1.tick_params(which = 'major',
#                 direction ='out', 
#                 length = 8, 
#                 width = 2, 
#                 labelsize = 11)
# plt.legend(title = 'Sensors', labels = ["LPS25", "BME280", "BMP388"])




# #
# # UV graph
# #
# dfUV = df
# dfUV['TimeOfFlight'] = pd.to_datetime(dfHum['TimeOfFlight'], unit = 's')
# plt.figure(figsize = (16,9))
# sns.set_palette(pallete3)
# sns.despine(top = True, offset = 40, trim = True)
# ax1 = sns.lineplot(x = 'TimeOfFlight', y = 'veml6070UV', data = dfUV)
# ax1.set_title('CanSat UV Sensor Readings', 
#                        fontproperties = prop,
#                        fontsize = 18)
# ax1.set_xlabel('Time of Flight', fontsize = 14)
# ax1.set_ylabel('UV [?]', fontsize = 14)
# ax1.tick_params(which = 'major',
#                 direction ='out', 
#                 length = 8, 
#                 width = 2, 
#                 labelsize = 11)
# plt.legend(title = 'Sensors', labels = ["VEML6075"])



#
# Pair Grid graphs
#

plt.figure(figsize = (16,9))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
#sns.kdeplot(df['bmp388Pres'], df['bme280Pres'], cmap=cmap, n_levels=60, shade=True)
ax2 = sns.jointplot(x = "bme280Pres", y = "mcp9808Temp", kind = 'kde', height = 8, space = 0, data = df)
#ax2 = ax2.plot(sns.regplot, sns.distplot)
ax2.ax_joint.set_xlabel("Pressure [hPa]")
ax2.ax_joint.set_ylabel("Temperature [°C]")
ax2.fig.suptitle('CanSat Pressure-Temperature Corelation', fontproperties = prop, fontsize = 18)
ax2.fig.subplots_adjust(top = 0.9)

plt.figure(figsize = (16,9))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
#sns.kdeplot(df_flight['Pressure'], df_flight['Temperature'], cmap=cmap, n_levels=60, shade=True)
ax2 = sns.jointplot(x = "bme280Pres", y = "mcp9808Temp", kind = 'kde', height = 8, space = 0, data = df_flight)
#ax2 = ax2.plot(sns.regplot, sns.distplot)
ax2.ax_joint.set_xlabel("Pressure [hPa]")
ax2.ax_joint.set_ylabel("Temperature [°C]")
ax2.fig.suptitle('CanSat Pressure-Temperature Corelation', fontproperties = prop, fontsize = 18)
ax2.fig.subplots_adjust(top = 0.9)


sns.set_palette(pallete4)
plt.figure(figsize = (15,9))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
#sns.kdeplot(df['bmp388Pres'], df['bme280Pres'], cmap=cmap, n_levels=60, shade=True)
ax2 = sns.jointplot(x = "mcp9808Temp", y = "bmp388Pres", kind = 'kde', height = 8, space = 0, data = df)
#ax2 = ax2.plot(sns.regplot, sns.distplot)
ax2.ax_joint.set_xlabel("Temperature [°C]")
ax2.ax_joint.set_ylabel("Pressure [hPa]")
ax2.fig.suptitle('CanSat 2021 Temperature-Pressure Corelation (flight time)', fontproperties = prop, fontsize = 18)
ax2.fig.subplots_adjust(top = 0.9)

plt.figure(figsize = (15,9))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
#sns.kdeplot(df_flight['Pressure'], df_flight['Temperature'], cmap=cmap, n_levels=60, shade=True)
ax2 = sns.jointplot(x = "mcp9808Temp", y = "bmp388Pres", kind = 'kde', height = 8, space = 0, data = df_flight)
#ax2 = ax2.plot(sns.regplot, sns.distplot)
ax2.ax_joint.set_xlabel("Temperature [°C]")
ax2.ax_joint.set_ylabel("Pressure [hPa]")
ax2.fig.suptitle('CanSat 2021 Temperature-Pressure Corelation (flight time)', fontproperties = prop, fontsize = 18)
ax2.fig.subplots_adjust(top = 0.9)


# plt.figure(figsize = (16,9))
# cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
# #sns.kdeplot(df['bmp388Pres'], df['bme280Pres'], cmap=cmap, n_levels=60, shade=True)
# ax2 = sns.jointplot(x = "bmp388Pres", y = "lps25Alt", kind = 'kde', height = 8, space = 0, data = df)
# #ax2 = ax2.plot(sns.regplot, sns.distplot)
# ax2.ax_joint.set_xlabel("BMP388 Pressure [mb]")
# ax2.ax_joint.set_ylabel("LPS25H Altitude [m]")
# ax2.fig.suptitle('CanSat Pressure-Altitude Corelation', fontproperties = prop, fontsize = 18)
# ax2.fig.subplots_adjust(top = 0.9)



# # dfAlt = pd.melt(df, id_vars = 'TimeOfFlight', value_vars = ["lps25Alt", "bme280Alt", "bmp388Alt"])
# # dfAlt['TimeOfFlight'] = pd.to_datetime(dfAlt['TimeOfFlight'], unit = 's')
# # print(dfAlt)

# # fig_grid = plt.figure(figsize = (16,9))
# # gs = fig_grid.add_gridspec(3,3)
# # ax1 = fig_grid.ass_subplot(gs)

# # fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16, ax17, ax18)) = plt.subplots(3, 6, sharex=True)
# # sns.set_palette(pallete3)
# # sns.despine(top = True, offset = 40, trim = True)

# # sns.kdeplot(df['bmp388Pres'], df['bmp388Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax1)
# # sns.kdeplot(df['bmp388Pres'], df['bme280Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax2)
# # sns.kdeplot(df['bmp388Pres'], df['mcp9808Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax3)
# # sns.kdeplot(df['bmp388Pres'], df['lis3Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax4)
# # sns.kdeplot(df['bmp388Pres'], df['lsm6dTemp'], cmap=cmap, n_levels=60, shade=True, ax = ax5)
# # sns.kdeplot(df['bmp388Pres'], df['lps25Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax6)
# # sns.kdeplot(df['bme280Pres'], df['bmp388Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax7)
# # sns.kdeplot(df['bme280Pres'], df['bme280Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax8)
# # sns.kdeplot(df['bme280Pres'], df['mcp9808Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax9)
# # sns.kdeplot(df['bme280Pres'], df['lis3Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax10)
# # sns.kdeplot(df['bme280Pres'], df['lsm6dTemp'], cmap=cmap, n_levels=60, shade=True, ax = ax11)
# # sns.kdeplot(df['bme280Pres'], df['lps25Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax12)
# # sns.kdeplot(df['lps25mBar'], df['bmp388Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax13)
# # sns.kdeplot(df['lps25mBar'], df['bme280Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax14)
# # sns.kdeplot(df['lps25mBar'], df['mcp9808Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax15)
# # sns.kdeplot(df['lps25mBar'], df['lis3Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax16)
# # sns.kdeplot(df['lps25mBar'], df['lsm6dTemp'], cmap=cmap, n_levels=60, shade=True, ax = ax17)
# # sns.kdeplot(df['lps25mBar'], df['lps25Temp'], cmap=cmap, n_levels=60, shade=True, ax = ax18)
# # # sns.jointplot(x = "bmp388Pres", y = "bme280Pres", kind = 'kde', height = 9, space = 0, data = df, ax = ax2)
# # # sns.jointplot(x = "bmp388Pres", y = "bmp388Temp", kind = 'hex', data = df, ax = ax3)
# # # sns.jointplot(x = "bmp388Pres", y = "bmp388Temp", kind = 'hex', data = df, ax = ax4)

# # # ax1.set_title('CanSat Altitude Sensor Readings', 
# # #                        fontproperties = prop,
# # #                        fontsize = 18)
# # # ax1.set_xlabel('Time of Flight', fontsize = 14)
# # # ax1.set_ylabel('Altitude [m]', fontsize = 14)
# # # ax1.tick_params(which = 'major',
# # #                 direction ='out', 
# # #                 length = 8, 
# # #                 width = 2, 
# # #                 labelsize = 11)
# # plt.legend(title = 'Sensors', labels = ["LPS25", "BME280", "BMP388"])



# # plt.figure(figsize = (16,9))
# # sns.despine(offset = 10, trim = True)
# # pltTemp = sns.jointplot(x = "bmp388Pres", y = "bmp388Temp", kind = 'hex', data = df)
# # #pltTemp.set(xlabel = 'Time', ylabel = 'Temperature [°C]')
# # #pltTemp.axes.set_title("CanSat temperature sensor readings",fontsize = 18)
# # #pltTemp.set_xlabel('Time',fontsize = 14)
# # #pltTemp.set_ylabel('Temperature [°C]')#,fontsize = 20)
# # #pltTemp.tick_params(labelsize = 5)
# # #plt.legend(title = 'Temperatures', labels = ["LPS25", "LSM6D", "LIS3", "MCP9808", "BME280", "BMP388"])



# # plt.figure(figsize = (16,9))
# # sns.despine(offset = 10, trim = True)
# # pltTemp = sns.jointplot(x = "bmp388Pres", y = "bme280Pres", kind = 'kde', height = 9, space = 0, data = df)
# # #pltTemp.set(xlabel = 'Time', ylabel = 'Temperature [°C]')
# # #pltTemp.axes.set_title("CanSat temperature sensor readings",fontsize = 18)
# # #pltTemp.set_xlabel('Time',fontsize = 14)
# # #pltTemp.set_ylabel('Temperature [°C]')#,fontsize = 20)
# # #pltTemp.tick_params(labelsize = 5)
# # #plt.legend(title = 'Temperatures', labels = ["LPS25", "LSM6D", "LIS3", "MCP9808", "BME280", "BMP388"])

plt.show()