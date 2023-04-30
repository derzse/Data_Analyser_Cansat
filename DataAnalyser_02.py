# -*- coding: utf-8 -*-
# Import all libraries needed for the tutorial
import re
import pandas as pd
import numpy as np
from numpy import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.font_manager as font_manager
#from mpl_toolkits.basemap import Basemap
import seaborn as sns
from dateutil import parser
import datetime

font_prop = font_manager.FontProperties(size=7)
grid_size=(1,3)
pallete1 = sns.color_palette("RdPu_r", n_colors = 8)
pallete2 = sns.color_palette("YlOrRd_r", n_colors = 5)
pallete3 = sns.color_palette("GnBu_r", n_colors = 5)

gsm_packeges = np.array([38, 53, 68, 97, 111, 126, 141, 156, 171, 186, 201, 216, 232, 262, 291, 305, 320, 336, 351, 367, 382, 398, 428, 442, 458, 487, 708, 765, 807, 882, 1041, 1055, 1070, 1161, 1191, 1251,  1265])


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

dfgsm = pd.read_csv('data_simcard.csv', sep=';')
firstRuntime = dfgsm['Runtime'].iloc[0]
dfgsm['TimeOfFlight'] = launchtime + dfgsm['Runtime'] - firstRuntime
dfgsm['bmp388Pres'] = 100*dfgsm['bmp388Pres']
#print(dfgsm)
dfgsm.index = pd.to_datetime(dfgsm['TimeOfFlight'], unit = 's')



time = df['Runtime']
temp = df['mcp9808Temp']
pres = df['bmp388Pres']
hum  = df['bmp388Alt']

timen = df['Runtime']
secn =   pd.to_datetime(df['TimeOfFlight'], unit = 's')
tempn = df['mcp9808Temp']
tempn1 = df['bmp388Temp']
presn = df['bmp388Pres']
humn  = df['bmp388Alt']


print(secn)
print(timen)

df_flight = df.iloc[389:430,:]
print("selection:")
print(df_flight)
time_f = df_flight['Runtime']
sec_f =   pd.to_datetime(df_flight['TimeOfFlight'], unit = 's')
temp_f = df_flight['mcp9808Temp']
pres_f = df_flight['bmp388Pres']
hum_f  = df_flight['bmp388Alt']



# flight window
x0 = 1838.72 # package 391
x1 = 2037.61 # package 431

#plotting the two lines for the flight time window
p1 = plt.axvline(x=x0,color='#EF9A9A')
p2 = plt.axvline(x=x1,color='#EF9A9A')


##### Combo of Temperature, Pressure and Humidity initial graphs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,5), sharex='col') #, sharey='row')
plt.subplots_adjust(wspace=0.3)
fig.suptitle('Cansat 2020 Initial Measurements')
hours = mdates.MinuteLocator(interval = 10)
h_fmt = mdates.DateFormatter('%H:%M')

##### Temperature graphs #####
ax1.plot(time, temp, color='dodgerblue', linewidth=1)

for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontproperties(font_prop)
ax1.xaxis.set_major_locator(hours)
ax1.xaxis.set_major_formatter(h_fmt)
ax1.xaxis.set_tick_params(rotation=0)
ax1.set_xlabel('Time [hh:mm]')
ax1.set_ylabel('Temperature [°C]')
#ax1.set_title('Air Temperature Measurements')
ax1.grid(True)

##### Pressure graphs #####
ax2.plot(time, pres, color='darkmagenta', linewidth=1)
#ax2 = sns.lineplot(x = 'TimeOfFlight', y = 'Temperature', data = dfn)

for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontproperties(font_prop)
ax2.xaxis.set_major_locator(hours)
ax2.xaxis.set_major_formatter(h_fmt)
ax2.xaxis.set_tick_params(rotation=0)
ax2.set_xlabel('Time [hh:mm]')
ax2.set_ylabel('Pressure [hPa]')
#ax2.set_title('Air Pressure Measurements')
ax2.grid(True)


##### Humidity graphs #####
#ax3.plot(time, hum, color='mediumvioletred', linewidth=1)
ax3.plot(time, hum, color='red', linewidth=1)

for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    label.set_fontproperties(font_prop)
ax3.xaxis.set_major_locator(hours)
ax3.xaxis.set_major_formatter(h_fmt)
ax3.xaxis.set_tick_params(rotation=0)
ax3.set_xlabel('Time [hh:mm]')
ax3.set_ylabel('Altitude [m]')
#ax3.set_title('Air Humidity Measurements')
ax3.grid(True)


##### Combo of Temperature, Pressure and Humidity corrected graphs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,5), sharex='col') #, sharey='row')
plt.subplots_adjust(wspace=0.3)
fig.suptitle('Cansat 2021 Sensor Readings (Temperature, Pressure, Altitude)')
minutes = mdates.SecondLocator(interval = 300)
m_fmt = mdates.DateFormatter('%H:%M:%S')
print(timen)

##### Temperature graphs #####
ax1.plot(secn, tempn, color='dodgerblue', linewidth=1.25)
# ax1.plot(secn, tempn1, color='royalblue', linewidth=1)
print(timen[0], type(timen[0]))
print(secn[0], type(secn[0]))

for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontproperties(font_prop)
#ax1.xaxis.set_major_locator(minutes)
ax1.xaxis.set_major_formatter(m_fmt)
ax1.xaxis.set_tick_params(rotation=0)
ax1.set_xlabel('Time [hh:mm:ss]')
ax1.set_ylabel('Temperature [°C]')
#ax1.set_title('Air Temperature Measurements')
ax1.grid(True)

##### Pressure graphs #####
ax2.plot(secn, presn, color='darkmagenta', linewidth=1.25)
#ax2 = sns.lineplot(x = 'TimeOfFlight', y = 'Temperature', data = dfn)

for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontproperties(font_prop)
#ax2.xaxis.set_major_locator(minutes)
ax2.xaxis.set_major_formatter(m_fmt)
ax2.xaxis.set_tick_params(rotation=0)
ax2.set_xlabel('Time [hh:mm:ss]')
ax2.set_ylabel('Pressure [hPa]')
#ax2.set_title('Air Pressure Measurements')
ax2.grid(True)


##### Humidity graphs #####
ax3.plot(secn, humn, color='mediumvioletred', linewidth=1.25)
ax3.plot(secn, humn, color='red', linewidth=1)

for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    label.set_fontproperties(font_prop)
#ax3.xaxis.set_major_locator(minutes)
ax3.xaxis.set_major_formatter(m_fmt)
ax3.xaxis.set_tick_params(rotation=0)
ax3.set_xlabel('Time [hh:mm:ss]')
ax3.set_ylabel('Altitude [m]')
#ax3.set_title('Air Humidity Measurements')
ax3.grid(True)



font_prop = font_manager.FontProperties(size=7)
##### Combo of Temperature, Pressure and Humidity corrected graphs in a timeframe
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,5), sharex='col') #, sharey='row')
plt.subplots_adjust(wspace=0.3)
fig.suptitle('Cansat 2021 Sensor Readings from 00:30:00 to 00:35:00 (mission time)')
minutes = mdates.SecondLocator(interval = 300)
m_fmt = mdates.DateFormatter('%H:%M:%S')
print(timen)

##### Temperature graphs #####
ax1.plot(sec_f, temp_f, color='dodgerblue', linewidth=1.25)

for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontproperties(font_prop)
#ax1.xaxis.set_major_locator(minutes)
ax1.xaxis.set_major_formatter(m_fmt)
ax1.xaxis.set_tick_params(rotation=0)
ax1.set_xlabel('Time [hh:mm:ss]')
ax1.set_ylabel('Temperature [°C]')
#ax1.set_title('Air Temperature Measurements')
ax1.grid(True)

##### Pressure graphs #####
ax2.plot(sec_f, pres_f, color='darkmagenta', linewidth=1.25)
#ax2 = sns.lineplot(x = 'TimeOfFlight', y = 'Temperature', data = dfn)

for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontproperties(font_prop)
#ax2.xaxis.set_major_locator(minutes)
ax2.xaxis.set_major_formatter(m_fmt)
ax2.xaxis.set_tick_params(rotation=0)
ax2.set_xlabel('Time [hh:mm:ss]')
ax2.set_ylabel('Pressure [hPa]')
#ax2.set_title('Air Pressure Measurements')
ax2.grid(True)


##### Humidity graphs #####
#ax3.plot(sec_f, hum_f, color='mediumvioletred', linewidth=1)
ax3.plot(sec_f, hum_f, color='red', linewidth=1.25)

for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    label.set_fontproperties(font_prop)
#ax3.xaxis.set_major_locator(minutes)
ax3.xaxis.set_major_formatter(m_fmt)
ax3.xaxis.set_tick_params(rotation=0)
ax3.set_xlabel('Time [hh:mm:ss]')
ax3.set_ylabel('Altitude [m]')
#ax3.set_title('Air Humidity Measurements')
ax3.grid(True)

font_prop = font_manager.FontProperties(size=10)


plt.show()

# fig2, ax4 = plt.subplots(figsize=(16,5))
# plt.subplots_adjust(right=0.75)

# ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
# ax6 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
# offset = 60
# #new_fixed_axis = ax6.get_grid_helper().new_fixed_axis
# #ax6.axis["right"] = new_fixed_axis(loc="right", axes=ax6,
# #                                        offset=(offset, 0))

# ax6.axis["right"].set_position(('outward', 60))


# color = 'dodgerblue'
# ax4.set_xlabel('Time [mm:ss]')
# ax4.set_ylabel('Temperature [°C]', color=color)
# ax4.plot(time_f, temp_f, color=color)
# ax4.tick_params(axis='y', labelcolor=color)

# color = 'darkmagenta'
# ax5.set_ylabel('Pressure [hPa]', color=color)  # we already handled the x-label with ax1
# ax5.plot(time_f, pres_f, color=color)
# ax5.tick_params(axis='y', labelcolor=color)

# color = 'red'
# ax6.set_ylabel('Pressure [hPa]', color=color)  # we already handled the x-label with ax1
# ax6.plot(time_f, hum_f, color=color)
# ax6.tick_params(axis='y', labelcolor=color)

# fig.tight_layout() 

plt.figure(figsize=(16,5))
#plt.title('Analog Cansat 2021 Corrected Measurements fom 00:10:00 to 00:15:00 (mission time)')
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)
plt.title('CanSat 2021 Sensor Readings from 13:35 to 13:40 (00:30:00 to 00:35:00 in mission time)')

color1 = 'dodgerblue'
color2 = 'darkmagenta'
color3 = 'red'

par1 = host.twinx()
par2 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                        offset=(offset, 0))
# print(type(par2))
# xlim = par2.axis.get_xlim()
# print(xlim)
# # ax1.axvspan(xlim[0], x0, color='#EF9A9A', alpha=0.5)
# par2.axvspan(x0, x1, color='#EF9A9A', alpha=0.5)
# #reset xlim
# par2.set_xlim(xlim)

par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

host.set_xlabel('Time [hh:mm:ss]')
host.set_ylabel('Temperature [°C]', color=color1)
par1.set_ylabel('Pressure [hPa]', color=color2)
par2.set_ylabel('Altitude [m]', color=color3)
host.xaxis.set_major_formatter(m_fmt)

p1, = host.plot(sec_f, temp_f, color=color1)
p2, = par1.plot(sec_f, pres_f, color=color2)
p3, = par2.plot(sec_f, hum_f, color=color3)

#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.draw()
plt.show()



# fig1, (ax4, ax5) = plt.subplots(1, 2, sharex='col')
# fig1.suptitle('Temperature-Humidity and Temperature-Pressure Distributions')

# counts, xedges, yedges, im = ax4.hist2d(np.isfinite(temp_f).values, np.isfinite(hum_f).values, bins=100, cmap='gist_heat')
# ax4.set_xlabel('Temperature [°C]')
# ax4.set_ylabel('Humidity [%]')
# plt.colorbar(im, label='Records', ax = ax4)

# print(np.isfinite(temp_f))

# counts, xedges, yedges, im = ax5.hist2d(temp_f.dropna(), pres_f.dropna(), bins=100, cmap='gist_heat')
# ax5.set_xlabel('Temperature [°C]')
# ax5.set_xlabel('Pressure [hPa]')
# plt.colorbar(im, label='Records', ax = ax5)
# plt.show()

# ##### Maps #####
# lat = df['Latitude'].values
# lon = df['Longitude'].values
# compass = df['Compass'].values

# def dms2dd(degrees, minutes, seconds, direction):
#     dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
#     if direction == 'S' or direction == 'W':
#         dd *= -1
#     return dd;

# def dd2dms(deg):
#     d = int(deg)
#     md = abs(deg - d) * 60
#     m = int(md)
#     sd = (md - m) * 60
#     return [d, m, sd]

# def parse_dms(dms):
#     parts = re.split('[^\d\w]+', dms)
#     d = int(parts[1])
#     m = int(parts[2])
#     s = int(parts[3])+int(parts[4])/10
#     ew = parts[5]
#     value = dms2dd(d,m,s,ew)
    
#     return value

# lat_dd = [parse_dms(l) for l in lat]
# lon_dd = [parse_dms(l) for l in lon]

# wdth = round(np.max(lon_dd)-np.min(lon_dd))
# hght = round(np.max(lat_dd)-np.min(lat_dd))
# print(wdth,hght)

# # 1. Draw the map background
# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='cyl', resolution='h', 
# 	llcrnrlat=np.min(lat_dd), urcrnrlat=np.max(lat_dd),
# 	llcrnrlon=np.min(lon_dd), urcrnrlon=np.max(lon_dd), 
# 	lat_0=np.mean(lat_dd), lon_0=np.mean(lon_dd),
#     #        width=10000000, height=10000000)
#     lat_ts=20)
# m.shadedrelief()
# m.drawcoastlines(color='gray')
# m.drawcountries(color='gray')
# m.drawstates(color='gray')

# # 2. scatter city data, with color reflecting population
# # and size reflecting area
# #x, y = m(lon_dd, lat_dd) 
# m.scatter(lon_dd, lat_dd, latlon=True, s=1, 
# 	marker='.', color='red',
# 	cmap='Reds', alpha=0.9)

# # 3. create colorbar and legend
# #plt.colorbar(label=r'$\log_{10}({\rm population})$')
# #plt.clim(3, 7)

# # make legend with dummy points
# #for a in [100, 300, 500]:
# #    plt.scatter([], [], c='k', alpha=0.5, s=a,
# #                label=str(a) + ' km$^2$')
# #plt.legend(scatterpoints=1, frameon=False,
#            #labelspacing=1, loc='lower left');
plt.show()