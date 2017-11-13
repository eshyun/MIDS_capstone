import lstm2

assert(lstm2.quarters_to_date(2011, 'Q1') == '2011-05-01')
assert(lstm2.quarters_to_date(2011, 'Q2') == '2011-08-01')
assert(lstm2.quarters_to_date(2011, 'Q3') == '2011-11-01')
assert(lstm2.quarters_to_date(2011, 'Q4') == '2012-02-01')
assert(lstm2.quarters_to_date(2011, 'FY') == '2012-06-01')
