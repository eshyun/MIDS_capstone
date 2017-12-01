from app import app
from flask import Flask, render_template, flash, redirect
from flask_wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
from .forms import LoginForm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

import sys
sys.path.append('../code/')
import lstm2

def generate_png(predicted_df, stock, days):
	ax = predicted_df[stock].plot(title=stock)
	#print(predicted_df[stock])
	column = str(days) + '-day prediction'
	mean = predicted_df[stock][column].mean()
	std = predicted_df[stock][column].std()
	ax.axhline(y=mean,color = 'k', ls='--', lw=0.5, label=column + ' mean')
	ax.axhline(y=mean+std,color = 'b', ls='-.', lw=0.5, label=column + ' + std')
	ax.axhline(y=mean-std,color = 'b', ls='-.', lw=0.5, label=column + ' - std')
	ax.set_xlabel('Days')
	ax.set_ylabel('Price')
	ax.legend()

	fig = ax.get_figure()
	stock_png = 'static/%s_%s.png' % (stock, days)
	fig.savefig('app/' + stock_png)
	print('generate_png', stock_png)
	plt.close(fig)

	return '/' + stock_png

def get_image_key(stock, days):
	return "%s_%s" % (stock,days)

def run_model(days, risk):
	'''
	Hook this into the algorithm to return the predicitons associated with given amount, returns and tolerance
	inputs: the amount ($), 30/45/60 for the days, and low/medium/high for risk level
	outputs: recommended and alternative stock pick, graphs for each of them
	'''
	#temporary example code, change to fill these with actual values based on model prediction
	print(days, risk)

	summary_df, predicted_df = lstm2.recommend_stocks(days, risk)
	print(summary_df)
	stock1 = summary_df['Stock Model'].iloc[0]
	rmse1 = summary_df['rsme'].iloc[0]
	stock2 = summary_df['Stock Model'].iloc[1]
	rmse2 = summary_df['rsme'].iloc[1]

	top_10 = summary_df.head(10)
	images = {}

	# Generates all images
	for index, row in top_10.iterrows():
		stock = row['Stock Model']
		key = get_image_key(stock, days)
		images[key] = generate_png(predicted_df, stock, days)

	png1 = images[get_image_key(stock1, days)]
	png2 = images[get_image_key(stock2, days)]

	print(stock1, stock2, png1, png2)
	return stock1, rmse1, png1, stock2, rmse2, png2, top_10


@app.route('/', methods=['POST', 'GET'])
def test():
	form = LoginForm()
	if form.validate_on_submit():
		stock, stock_rmse, stock_plot, alt, alt_rmse, alt_plot,summary_df = run_model(int(form.date.data),
													 form.risk_tolerance.data)
		return render_template('recommendation.html',form=form,
			date = str(form.date.data), risk = form.risk_tolerance.data, main_rmse = stock_rmse,
			plot_name=stock_plot, alternative_plot_name=alt_plot, recommended = stock, alternate = alt, alternate_rmse = alt_rmse,
			dataframe=summary_df.to_html(index=False),
			df_json=summary_df.reset_index().to_json(orient='records'))

	print(form.date.data, form.risk_tolerance.data)
	return render_template('recommendation.html',form=form, date = "__", risk = "__", main_rmse = '__',
			plot_name="/static/question.png", 
			alternative_plot_name="/static/question.png",
			alternate_rmse = '__',
			dataframe=None, df_json=None
			)









