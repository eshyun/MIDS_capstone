from app import app
from flask import Flask, render_template, flash, redirect
from flask_wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
from .forms import LoginForm
#import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../code/')
import lstm2

def generate_png(predicted_df, stock, days):
	ax = predicted_df[stock].plot(title=stock)
	print(predicted_df[stock])
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
	return '/' + stock_png

def run_model(amount, days, risk):
	'''
	Hook this into the algorithm to return the predicitons associated with given amount, returns and tolerance
	inputs: the amount ($), 30/45/60 for the days, and low/medium/high for risk level
	outputs: recommended and alternative stock pick, graphs for each of them
	'''
	#temporary example code, change to fill these with actual values based on model prediction
	print(amount, days, risk)

	summary_df, predicted_df = lstm2.recommend_stocks(days, risk)
	stock1 = summary_df['Stock Model'].iloc[0]
	rmse1 = summary_df['rsme'].iloc[0]
	stock2 = summary_df['Stock Model'].iloc[1]
	rmse2 = summary_df['rsme'].iloc[1]

	png1 = generate_png(predicted_df, stock1, days)
	png2 = generate_png(predicted_df, stock2, days)

	print(stock1, stock2, png1, png2)
	print(summary_df)
	return stock1, rmse1, png1, stock2, rmse2, png2, summary_df


@app.route('/', methods=['POST', 'GET'])
def test():
	form = LoginForm()
	if form.validate_on_submit():
		stock, stock_rmse, stock_plot, alt, alt_rmse, alt_plot,summary_df = run_model(form.amount.data, int(form.date.data),
													 form.risk_tolerance.data)
		return render_template('recommendation.html',form=form, amount = form.amount.data,
			date = str(form.date.data), risk = form.risk_tolerance.data, main_rmse = stock_rmse,
			plot_name=stock_plot, alternative_plot_name=alt_plot, recommended = stock, alternate = alt, alternate_rmse = alt_rmse,
			dataframe=summary_df.head(10).to_html(index=False))

	print(form.amount.data, form.date.data, form.risk_tolerance.data)
	return render_template('recommendation.html',form=form, amount = "___", date = "__", risk = "__", main_rmse = '__',
			plot_name="/static/question.png", 
			alternative_plot_name="/static/question.png",
			alternate_rmse = '__',
			dataframe=None
			)









