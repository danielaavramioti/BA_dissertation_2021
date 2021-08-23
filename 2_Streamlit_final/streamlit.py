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

def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()

st.subheader('Channel response curves')




# =============================================================================
# #this runs correctly
# @st.cache
# def load_data():
#     df=pd.read_excel('all_results_2.xlsx', sheet_name='PVID & BVOD')
#     return df
# 
# df=load_data()
# df
# =============================================================================

#st.line_chart(df)

st.sidebar.subheader('Channel options')
#Sidebar options
params={
        'Channels': st.sidebar.multiselect('Channels', ('TV 2017', 'TV 2018','B&D', 'PVID & BVOD', 
                                                        'FB', 'Radio', 'Cinema', 'Youtube', 'Print'))
        }
#channels=st.sidebar.multiselect('Channels', ('B&D', 'PVID & BVOD', 'FB', 'Radio', 'Cinema', 'Youtube', 'Print'))

#channel_list=params['Channels'].copy()


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

#this works but it overwrites
def get_curves(channel):
    
    df=load_data(channel)
    ax.plot(df['Raw Spend'],df['Revenue Contribution'], label=f"{channel} response curve")

    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax.set_title('Media channels Response curve');
    
    return 

#plotting all the curves from the selection pane
for channel in params['Channels']:
    df=load_data(channel)
    #st.subheader(f'{channel} response dataframe')

    fig=get_curves(channel)
    
#st.pyplot(fig)



#fit the functions for each channel
all_functions={}
for channel in params['Channels']:
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
    
    ax2.plot(x, all_functions[channel](x), label=channel)
    #ax2.plot(x,(np.array(all_functions[channel](x))/x), label=f'ROI {channel}')
                # Put a legend to the right of the current axis
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax2.set_title('Media channels Response curve');
    return

for channel in params['Channels']:
    fig2=plot_fitted_funcs(channel)
            
st.pyplot(fig2)
        


#Plot the ROi for each channel
st.subheader('ROI plots')
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
def plot_ROI_funcs(channel):  
    
    df=load_data(channel)
    ax3.plot(df['Raw Spend'],(df['Revenue Contribution']/df['Raw Spend']), label=f"{channel} ROI")
        
    #ax2.plot(x,(all_functions[channel](x)/x), label=f'ROI {channel}')
                # Put a legend to the right of the current axis
    ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    ax3.set_title('Media channels ROI');
    return

for channel in params['Channels']:
    fig3=plot_ROI_funcs(channel)
            
st.pyplot(fig3)

# all_functions={'Youtube': 'youtube_func', 'B&D':'BD_func', 'Cinema': 'cinema_func', 'Radio':'radio_func', 
#                 'FB': 'FB_func', 'PVID & BVOD': 'PVID_BVOD_func', 'Print':'print_func', 
#                 'TV 2017':'TV_2017_func', 'TV 2018':'TV_2018_func'}



#creating slider for defining bounds for each channel
all_channel_bounds={}
st.sidebar.subheader('Input your channel bounds')
for channel in params['Channels']:
    channel_budget = st.sidebar.slider(f'Select a budget range for {channel}', value=[0,1000000])
    all_channel_bounds[channel]=channel_budget
    st.sidebar.write(f'You selected a budget between {channel_budget[0]} and {channel_budget[1]} for {channel}')




# #generate the arrays for each channel spend
# def total_revenue(all_spend_arrays):
#     sum_revenue=0
#     for channel in params['Channels']:
#         sum_revenue+=all_functions[channel](all_spend_arrays[channel])
#     return -(sum_revenue)


# def func1(x):
#     def func2(y):
#         return x+y
#     return func2

# a=func1(3)
# a(2)

# #selected_channels=params['Channels']

def create_revenue_func(selected_channels):
    def new_revenue_func(input_array):
        total=0
        for i in range(len(selected_channels)):
            channel=selected_channels[i]
            f=all_functions[channel]
            total+=f(input_array[i])
        return -1*(total)
    return new_revenue_func




# =============================================================================
# #maximisie ROI?
# #ROI=total-input_array
# def create_revenue_func(selected_channels):
#     def new_revenue_func(input_array):
#         total=0
#         for i in range(len(selected_channels)):
#             channel=selected_channels[i]
#             f=all_functions[channel]
#             total+=f(input_array[i])
#         
#         ROI=total-sum(input_array)
#         return -1*(ROI)
#     return new_revenue_func
# 
# =============================================================================



revenue_func=create_revenue_func(params['Channels'])
starting_values=[5 for i in range(len(params['Channels']))]
all_channel_bounds_2=[all_channel_bounds[i] for i in params['Channels']]
print(revenue_func)
print(starting_values)
print(all_channel_bounds)




#request a budget input
budget = st.sidebar.number_input('Enter your total budget constraint')

#defining constraints for the optimiser
def budget_constraint(input_array):
    """constrain budget to always be >= spend"""
    return budget - sum(input_array)

def ineq_constraint(x):
    """constrain all elements of x to be >= 0"""
    return x

#testing this one
def revenue_constraint(input_array):
    """constrain max revenue to always be >= above budget constraint"""
    return budget - revenue_func(input_array)

#creating the dictionary of constraints
constraints = [{
    'type' : 'ineq',
    'fun' : budget_constraint
},
    {'type': 'ineq', 
     'fun': ineq_constraint
},
     {'type': 'ineq', 
     'fun': revenue_constraint
}]

#defining options which wil be used for setting maximum iterations
options = {'maxiter' : 10000}

#FIX THE SPEND ARRAY ISSUE
#the optimisation itself
result = minimize(
    fun = revenue_func,
    x0 = starting_values,
    bounds = all_channel_bounds_2,
    constraints = constraints,
    method = 'SLSQP',
    options=options
)
print(result)



#Show results of the optimiser
def summarise(result):
    for i in range(len(params['Channels'])):
        st.write(f"Optimal {params['Channels'][i]} budget = £{result.x[i]:.0f}")

    st.write(40*"-")
    st.write(f"Max revenue = £{-revenue_func(result.x):.0f}")
    st.write(40*"-")
    
summarise(result)

def create_pie_chart(result):
    data_pie_chart=[]
    labels=[]
    explosion_values=[]
    for i in range(len(params['Channels'])):
        #generate random values for the pie explosion:
        explosion_values.append(round(uniform(0, 0.4),1))
        #generate labels for the piechart
        labels.append(params['Channels'][i])
        #generate the data for the pie chart
        data_pie_chart.append(result.x[i])

    # Build the pie chart
    st.subheader("Budget split by channel")
    figureObject, axesObject = plotter.subplots()
    axesObject.pie(data_pie_chart,
                   #explode      = explosion_values,
                   labels       = labels,
                   autopct      = '%1.2f%%',
                   startangle   = 90)
    
     
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig_circle = plt.gcf()
    fig_circle.gca().add_artist(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    axesObject.axis('equal')

    st.pyplot(figureObject)

create_pie_chart(result)   

    
    