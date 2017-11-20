from flask_wtf import Form
from wtforms import DateField, IntegerField, RadioField
from wtforms.validators import DataRequired

class LoginForm(Form):
	amount = IntegerField("amount to invest",validators=[DataRequired()])
	date = RadioField('timeframe', choices = [('30', '30 days'), ('45', '45 days'), ('60', '60 days')], validators=[DataRequired()])
	risk_tolerance = RadioField( 'risk', choices = [('low', 'low'), ('medium', 'medium'), ('high', 'high')] , validators=[DataRequired()])