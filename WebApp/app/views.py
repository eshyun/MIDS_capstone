from app import app
from flask import Flask, render_template, flash, redirect
from flask_wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
from .forms import LoginForm
import matplotlib.pyplot as plt
import os

def run_model(amount, days, risk):
	'''
	Hook this into the algorithm to return the predicitons associated with given amount, returns and tolerance
	inputs: the amount ($), 30/45/60 for the days, and low/medium/high for risk level
	outputs: recommended and alternative stock pick, graphs for each of them
	'''
	#temporary example code, change to fill these with actual values based on model prediction
	recommended_stock = "Example_Stock"
	alternative_stock = "Example_Stock_2"
	plot_of_stock = "/static/MSFT_FORECAST.png"
	plot_of_alternative_stock = "/static/ORCL_FORECAST.png"
	return recommended_stock, alternative_stock, plot_of_stock, plot_of_alternative_stock


@app.route('/', methods=['POST', 'GET'])
def test():
	form = LoginForm()
	if form.validate_on_submit():
		stock, alt, stock_plot, alt_plot = run_model(str(form.amount.data), str(form.date.data),
													 str(form.risk_tolerance.data))
		return render_template('recommendation.html',form=form, amount = str(form.amount.data), 
			date = str(form.date.data), risk = form.risk_tolerance.data,
			plot_name=stock_plot, alternative_plot_name=alt_plot, recommended = stock, alternate = alt)
	return render_template('recommendation.html',form=form, amount = "___", date = "__", risk = "__",
			plot_name="/static/question.png", 
			alternative_plot_name="/static/question.png")









