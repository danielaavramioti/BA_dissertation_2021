import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import plotly.express as px
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import minimize
from nltk import flatten
from random import randint, uniform
import matplotlib.pyplot as plotter

st.title('Media budget optimisations')
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('deprecation.showPyplotGlobalUse', False)
interactive_charts=st.beta_container()

#status bar
def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()

st.subheader('Channel response curves')




#selection on the sidebar
st.sidebar.subheader('Channel options')
#Sidebar options
#new option
params={
        'Channels': st.sidebar.multiselect('Channels', ('TV', 'PVID & BVOD', 
                                                        'FB', 'Radio', 'Cinema', 'Youtube', 'Print'))}

#creating a copy so we can access the excel with the results without having to make changes to the selection made on the app
channel_list=params['Channels'].copy()
if 'TV' in channel_list:
    index = channel_list.index('TV')
    channel_list[index]='B&D'
else:
    pass

#test
@st.cache
#generate dataframe name for each channel so we can save results
def generate_df_per_channel_name(channel):

    name=f"df_{channel}"

    return name

def load_data(channel):
    
    name=generate_df_per_channel_name(channel)
    name=pd.read_excel('all_results_2.xlsx', sheet_name=f'{channel}')
    return name



#defining names for all channel functions
all_functions={'Youtube': 'youtube_func', 'B&D':'BD_func', 'Cinema': 'cinema_func', 'Radio':'radio_func', 
                'FB': 'FB_func', 'PVID & BVOD': 'PVID_BVOD_func', 'Print':'print_func', 
                'TV 2017':'TV_2017_func', 'TV 2018':'TV_2018_func'}



#must define the axes prior to plotting in order to be able to build curves ontop of one another
fig, ax = plt.subplots()
ax.set_xlabel('Raw values Spend (£)')
ax.set_xlim(0,1000000)

ax.set_ylabel('Channels contribution to revenue (£)')
#rotating x-axis ticks
ax.set_xticklabels(ax.get_xticks(),rotation=40)

#work on this
ax.ticklabel_format(style='plain', axis='y') #prevents scientific notation
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

#function for creating the response curves
def get_curves(channel):
    
    df=load_data(channel)
    ax.plot(df['Raw Spend'],df['Revenue Contribution'], label="['TV' if channel =='B&D' else channel for channel in channel_list] +'response curve'")

    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax.set_title('Media channels Response curve');
    
    return 

#plotting all the curves from the selection pane
for channel in channel_list:

    df=load_data(channel)
    #st.subheader(f'{channel} response dataframe')

    fig=get_curves(channel)
    
#st.pyplot(fig) #currently we are not showing this figure because we can actually just show the figure after fitting the functions with interpolation. The figures will be identical so the above code is useful for cross checking the results



#fit the functions for each channel
all_functions={}
for channel in channel_list:
    df=load_data(channel)
    if channel=='Youtube':
        youtube_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=youtube_func
    elif channel=='B&D':
        BD_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=BD_func
    elif channel=='Cinema':
        cinema_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=cinema_func
    elif channel=='Radio':
        radio_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=radio_func
    elif channel=='FB':
        FB_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=FB_func
    elif channel=='PVID & BVOD':
        PVID_BVOD_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=PVID_BVOD_func
    elif channel=='Print':
        print_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=print_func
    elif channel=='TV 2017':
        TV_2017_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=TV_2017_func
    elif channel=='TV 2018':
        TV_2018_func=interp1d(df['Raw Spend'],df['Revenue Contribution'], fill_value="extrapolate")
        all_functions[channel]=TV_2018_func
    #save results



#Plot the fitted functions that convert Spend to Revenue

fig2, ax2 = plt.subplots()
ax2.set_xlabel('Raw values Spend (£)')
ax2.set_xlim(0,1000000)

ax2.set_ylabel('Channels contribution to revenue (£)')
#rotating x-axis ticks
ax2.set_xticklabels(ax.get_xticks(),rotation=40)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#work on this
ax2.ticklabel_format(style='plain', axis='y') #prevents scientific notation
x=range(1,1000000)
def plot_fitted_funcs(channel):  
    
    ax2.plot(x, all_functions[channel](x), label=['TV' if channel =='B&D' else channel][0])
    #ax2.plot(x,(np.array(all_functions[channel](x))/x), label=f'ROI {channel}')
                # Put a legend to the right of the current axis
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax2.set_title('Media channels Response curve');
    return

for channel in channel_list:
    fig2=plot_fitted_funcs(channel)
            
st.pyplot(fig2) #we show this on the actual web app
        


#Plot the ROI for each channel
#st.subheader('ROI plots') #we dont need to show this
fig3, ax3 = plt.subplots()
ax3.set_xlabel('Raw values Spend (£)')
ax3.set_xlim(0,1000000)

ax3.set_ylabel('Channels contribution to revenue (£)')
#rotating x-axis ticks
ax3.set_xticklabels(ax3.get_xticks(),rotation=40)
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#work on this
ax3.ticklabel_format(style='plain', axis='y') #prevents scientific notation
#x=range(1,1000000)
#function to plot the ROI
def plot_ROI_funcs(channel):  
    
    df=load_data(channel)
    ax3.plot(df['Raw Spend'],(df['Revenue Contribution']/df['Raw Spend']), label=['TV' if channel =='B&D' else channel][0] +" ROI")#we add the if function to show the TV label on the web app rather than B&D
        
    #ax2.plot(x,(all_functions[channel](x)/x), label=f'ROI {channel}')
                # Put a legend to the right of the current axis
    ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax3.set_title('Media channels ROI');
    return

#plot the ROI curves for each channel in channel list
for channel in channel_list:
    fig3=plot_ROI_funcs(channel)
            
#st.pyplot(fig3)#we don't need to show this on the actual website


campaign_length = st.sidebar.number_input("Input campaign length in number of weeks", min_value=0, max_value=52) #the value bounds can also change if desired

#creating slider for defining bounds for each channel
total_spend=0
total_return=0
st.sidebar.subheader('Input your desired channel budget')
for channel in channel_list:
    f=all_functions[channel]#get the function from the interpolation stage that is corresponding to the channel
    channel_budget = st.sidebar.number_input(f"Select a desired budget for {['TV' if channel =='B&D' else channel][0]}") #the value bounds can also change if desired
    total_spend+=channel_budget
    total_return+=f(channel_budget)
    #st.sidebar.write(f"You selected a budget between {channel_budget[0]} and {channel_budget[1]} for {['TV' if channel =='B&D' else channel][0]}")

st.write(f"Max off-trade value generation per week = £{total_return:.0f}")
st.write(f"Max off-trade value generation for £{total_spend:.0f} spent is £{(total_return*campaign_length):.0f}")
