#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:17:15 2024

@author: akshay
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import csv
#import pdflatex


df = pd.read_csv('./renpho_data/240224_Renpho Health-Akshay.csv', index_col=None, header=None)
df = df.iloc[1:] # drop column the labels
#df = df.iloc[:,:1] #drop the first column
df.columns = ['Date','Weight','BMI','BodyFat', 'FatFreeBodyWeight','SubcutFat','ViscFat','BodyWater',
              'SkelMuscle','MuscleMass','BoneMass','Protein','BMR','MetAge','Remarks']

columns_to_convert = df.columns[1:]  # Replace with your actual column names

for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')


#df['Date'] = pd.to_datetime(df['Date'], format='%b %d %Y at %H:%M:%S')
df['Date'] = pd.to_datetime(df['Date']).dt.date
#df.set_index('Date', inplace=True)  # Set 'date' as the index
#heath_std_df = pd.DataFrame(['Weight','BMI','BodyFat', 'FatFreeBodyWeight','SubcutFat','ViscFat','BodyWater',
 #             'SkelMuscle','MuscleMass','BoneMass','Protein','BMR'])
# Manually enter the weight range shown in the app #50.37
health_std = {'Weight': [55, 68.08],
     'BMI': [18.5, 25],
     'BodyFat':[13, 17],
     'SubcutFat':[8.6, 16.7],
     'ViscFat':[1,6],
     'BodyWater':[50,65],
     'SkelMuscle':[49,59],
     'MuscleMass':[44,52.40],
     'BoneMass':[1.86,3.10],
     'Protein':[16,20],
     'BMR':[1383,1600]}
#Input your goals here
health_goal = {'Weight': 59,
     'BMI': [18.5, 25],
     'BodyFat':17,
     'SubcutFat':16.7,
     'ViscFat':5,
     'BodyWater':[50,65],
     'SkelMuscle':[49,59],
     'MuscleMass':48,
     'BoneMass':[1.86,3.10],
     'Protein':20,
     'BMR':[1383,1600]}

class health_trend():
    def __init__(self,df, health_std, health_goal):
        self.df = df
        self.health_std = health_std
        self.health_goal = health_goal
        
    def predict_param(self, healthparam, goal):
        local_df = df.copy()  # Work on a local copy
        local_df.set_index('Date', inplace=True)  # Set 'date' as the index
        x =  np.arange(local_df.index.size )
        #self.power_law_fit(local_df[healthparam], x, goal)
        #fit = np.polyfit(x, df[healthparam], deg=1)
        start_val = local_df[healthparam].iloc[0]
        current_val = local_df[healthparam].iloc[-1]
        # Fit the model
        model = sm.OLS(df[healthparam], sm.add_constant(x)).fit() #add intercept term
        intercept, slope = model.params
        print(f"==============  {healthparam}  ===============")
        print (f"{healthparam} slope : " + str(round(slope,3)) + f" units of {healthparam} per day")
        #print ("Intercept : " + str(fit[1]))
        asof = str(df['Date'].iloc[-1])
        days2root = round((goal - intercept)/slope) - local_df.index.size
        rootDate =  local_df.index[-1] + datetime.timedelta(days2root)
       # rootDate = messages_per_day.index[-1] + datetime.timedelta(round(-fit[1]/fit[0],0))
        print(str(days2root) + f" more days until the {healthparam} goal of {goal} (on {rootDate})")
        

        # print(f"Intercept: {intercept}")
        # print(f"Slope: {slope}")
        #print(model.summary())
        r_squared = model.rsquared
        print(f"R-squared: {r_squared:0.3f}")
        
        # To get the p-values:
        p_values = model.pvalues[0]
        print(f"P-value: {p_values}")
        fg = plt.figure(figsize=(10, 6))
        plt.plot(local_df.index, df[healthparam], color = plt.cm.tab10(3), marker='o', linestyle = '-', alpha= 0.8)
        plt.grid(visible=1)
        plt.xlabel('Date')
        plt.ylabel(healthparam)
        plt.title(f'{healthparam} change over diet (as of {asof})')
       
        #plot the fit
        #fit_function = np.poly1d(fit)
        
        new_dates = [local_df.index[-1] + datetime.timedelta(days=x) for x in range(np.int64(days2root+2))]
        #new_dates = pd.date_range(start=local_df.index[-1] ,periods=days2root, freq='D')
        extended_index = local_df.index.union(new_dates)
        # If your model included an intercept, you need to add a constant to new_x
        #new_x_with_const = sm.add_constant(new_x)
        #predited_y = fit_function(np.arange(extended_index.size))
        # Use the model to make predictions
        predicted_y = model.predict(sm.add_constant(np.arange(extended_index.size)))
        #date_range = pd.date_range(start_date, periods=25, freq='D')
        plt.plot(extended_index, predicted_y, color = plt.cm.tab10(3), linestyle = 'dashed')
        plt.axhline(y=goal, color='g', linestyle='--',linewidth = 2)
        plt.fill_between(extended_index, self.health_std[healthparam][0], self.health_std[healthparam][1], color='green', alpha=0.2)  # alpha controls the transparency

        plt.text(extended_index[np.int64(extended_index.size/3)], goal+1, str(days2root) + f" more days until the {healthparam} goal of {goal} (on {rootDate})", fontsize=12)
        plt.text(0.5, 0.9, f'y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='center')
        plt.show()
        plt.close()
        return start_val, current_val, goal, slope, r_squared, p_values, days2root, rootDate, fg
    def health_stat_extractor(self):
        figs = []#np.zeros(len(self.health_std),1)
        values = []
        for healthparam in self.health_std.keys():
            goal = round(np.mean(self.health_goal[healthparam]) ,2)
            #*_,fg = self.predict_param(healthparam,goal)#fg is the last returned values
            *vals, fg = self.predict_param(healthparam, goal) #values lists all returned values except the last
            vals_rounded = [round(x, 3) if isinstance(x, float) else x for x in vals]
            figs.append(fg)
            values.append([healthparam] + vals_rounded)
        #print((values))
        self.stat_table = values
        self.figs = figs
        self.values = values
        return figs, values
        
        
    def summary_figure(self):
        summaryFileName = './renpho_figures/' + str(df['Date'].iloc[-1]) + '_renpho_prediction_figure.pdf'
        with PdfPages(summaryFileName) as pdf:
            for fig in self.figs:
                pdf.savefig(fig, bbox_inches='tight') 
        
   
        
       
    def summary_table(self):    
        pdf_file = './renpho_figures/' + str(df['Date'].iloc[-1]) + '_renpho_prediction_table.pdf'
        headings = ['Metric','start_val', 'current_val', 'Goal', 'slope', 'R_squared', 'P value', 'Days to go', 'Goal day']
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        pdf = SimpleDocTemplate(pdf_file, pagesize=letter)
        elements = []  # List to hold the elements of the PDF
        
        # Create a table with your data
        table_csv = [headings] + self.values
        table = Table([headings] + self.values)
        #print(table)
        def stringify_row(row):
            return [str(item) for item in row]
        csv_file_path = './renpho_figures/' + str(df['Date'].iloc[-1]) + '_renpho_summary.csv'
        
        # Writing to the CSV file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in table_csv:
                writer.writerow(stringify_row(row))
        self.summary = table
        # Create a title
        title = Paragraph("Summary of health stats: " + str(df['Date'].iloc[-1]), title_style)
        # Optionally, add some style to the table
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            # Add black lines around each cell
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Add a black line around the outer border of the table
            ('BOX', (0, 0), (-1, -1), 1, colors.black)
        ])
        table.setStyle(table_style)
        # Add the table to the elements of the PDF
        elements.append(table)
       
            
        elements = [title, Spacer(1, 12), table]  # 
        
        # Build the PDF
        pdf.build(elements)
        
        print(f"PDF file '{pdf_file}' created successfully.")
        
        # Specify the CSV file path
        
        
        print(f"Data saved to {csv_file_path}")
        return table
    def calculate_moving_average(self, data, window_size):
        return data.rolling(window=window_size).mean()
    ## To do
    ## Stair case fit function
    ## Power law fit function
    def power_law_fit(self, healthparam, goal):
        local_df = df.copy()  # Work on a local copy
        local_df.set_index('Date', inplace=True)  # Set 'date' as the index
        x =  np.arange(local_df.index.size )
        #self.power_law_fit(local_df[healthparam], x, goal)
        #fit = np.polyfit(x, df[healthparam], deg=1)
        start_val = local_df[healthparam].iloc[0]
        current_val = local_df[healthparam].iloc[-1]
        log_x = np.log(x)
        log_y = np.log(local_df[healthparam])
        results_power_law = sm.OLS(log_y, sm.add_constant(log_x)).fit()
        # Intercept (log(a)) and slope (b)
        a = np.exp(results_power_law.params[0])
        b = results_power_law.params[1]
        intercept, slope = results_power_law.params
        print(f"==============  {healthparam}  ===============")
        print (f"{healthparam} slope : " + str(round(slope,3)) + f" units of {healthparam} per day")
        #print ("Intercept : " + str(fit[1]))
        asof = str(df['Date'].iloc[-1])
        days2root = round((np.log(goal) - intercept)/slope) - local_df.index.size
        rootDate =  local_df.index[-1] + datetime.timedelta(days2root)
       # rootDate = messages_per_day.index[-1] + datetime.timedelta(round(-fit[1]/fit[0],0))
        print(str(days2root) + f" more days until the {healthparam} goal of {goal} (on {rootDate})")
        
        r_squared = results_power_law.rsquared
        print(f"R-squared: {r_squared:0.3f}")
        
        # To get the p-values:
        p_values = results_power_law.pvalues[0]
        print(f"P-value: {p_values}")
        predicted_log_y = results_power_law.predict(sm.add_constant(log_x))
        predicted_y = np.exp(predicted_log_y)

        fg = plt.figure(figsize=(10, 6))
        plt.plot(local_df.index, df[healthparam], color = plt.cm.tab10(3), marker='o', linestyle = '-', alpha= 0.8)
        plt.grid(visible=1)
        plt.xlabel('Date')
        plt.ylabel(healthparam)
        plt.title(f'{healthparam} change over diet (as of {asof})')
       
        #plot the fit
        #fit_function = np.poly1d(fit)
        
        new_dates = [local_df.index[-1] + datetime.timedelta(days=x) for x in range(np.int64(days2root+2))]
        #new_dates = pd.date_range(start=local_df.index[-1] ,periods=days2root, freq='D')
        extended_index = local_df.index.union(new_dates)
        # If your model included an intercept, you need to add a constant to new_x
        #new_x_with_const = sm.add_constant(new_x)
        #predited_y = fit_function(np.arange(extended_index.size))
        # Use the model to make predictions
        predicted_y = results_power_law.predict(sm.add_constant(np.arange(extended_index.size)))
        #date_range = pd.date_range(start_date, periods=25, freq='D')
        plt.plot(extended_index, predicted_y, color = plt.cm.tab10(3), linestyle = 'dashed')
        plt.axhline(y=goal, color='g', linestyle='--',linewidth = 2)
        plt.fill_between(extended_index, self.health_std[healthparam][0], self.health_std[healthparam][1], color='green', alpha=0.2)  # alpha controls the transparency
    
        plt.text(extended_index[np.int64(extended_index.size/3)], goal+1, str(days2root) + f" more days until the {healthparam} goal of {goal} (on {rootDate})", fontsize=12)
        plt.text(0.5, 0.9, f'y = {slope:.3f}x + {intercept:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='center')
        plt.show()
        plt.close()
        return start_val, current_val, goal, slope, r_squared, p_values, days2root, rootDate, fg

hp = health_trend(df,health_std,health_goal)
hp.health_stat_extractor()
hp.summary_figure()
hp.summary_table()
#_, fi = hp.predict_param('Weight', 60)
# bf = hp.predict_param('BodyFat', 17)
# sf = hp.predict_param('SubcutFat', 15)
# healthparam =  'BodyFat'
# goal = 17
# wd = hp.predict_param(healthparam, goal)
 # Example usage with dummy data
# values = [
#      (1, 2, -0.1, 0.95, 0.001, 10, "2024-01-01", "Figure 1"),
#      # Add more rows as needed
#  ]
        
# hp.summary_table(values)