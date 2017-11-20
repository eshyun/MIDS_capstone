from app import app
from flask import Flask, render_template, flash, redirect
from flask_wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
from .forms import LoginForm
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../code/')
import lstm2

def generate_png(predicted_df, stock):
	fig = predicted_df[stock].plot().get_figure()
	stock_png = 'static/' + stock + '.png'
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
	stock2 = summary_df['Stock Model'].iloc[1]
	png1 = generate_png(predicted_df, stock1)
	png2 = generate_png(predicted_df, stock2)

	print(stock1, stock2, png1, png2)
	return stock1, stock2, png1, png2


@app.route('/', methods=['POST', 'GET'])
def test():
	form = LoginForm()
	if form.validate_on_submit():
		stock, alt, stock_plot, alt_plot = run_model(form.amount.data, int(form.date.data),
													 form.risk_tolerance.data)
		return render_template('recommendation.html',form=form, amount = form.amount.data,
			date = str(form.date.data), risk = form.risk_tolerance.data,
			plot_name=stock_plot, alternative_plot_name=alt_plot, recommended = stock, alternate = alt)

	print(form.amount.data, form.date.data, form.risk_tolerance.data)
	return render_template('recommendation.html',form=form, amount = "___", date = "__", risk = "__",
			plot_name="/static/question.png", 
			alternative_plot_name="/static/question.png")









