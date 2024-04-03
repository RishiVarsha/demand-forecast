from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from keras.models import load_model

app = Flask(__name__)

# Load your preprocessed data
train = pd.read_csv('C:/Users/Rishi Varsha/Downloads/demand-forecasting-kernels-only/train.csv')
# Load the test dataset
test_df = pd.read_csv('C:/Users/Rishi Varsha/Downloads/demand-forecasting-kernels-only/test.csv')

# Load the submission dataset
submission_df = pd.read_csv('C:/Users/Rishi Varsha/Downloads/demand-forecasting-kernels-only/submission.csv')

# Combine the datasets using the 'id' column as the index
df = pd.concat([test_df.set_index('id'), submission_df.set_index('id')], axis=1).reset_index()

# Merge the train DataFrame with the combined DataFrame df
# Drop the 'id' column in the DataFrame df
df.drop(columns=['id'], inplace=True)

# Concatenate train and df
final_df = pd.concat([train, df], ignore_index=True)
# Convert the 'date' column to Pandas Timestamp objects
final_df['date'] = pd.to_datetime(final_df['date'])

model_lstm = load_model("C:/Users/Rishi Varsha/Downloads/lstm_model_with_mape.h5")
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Get user input from the form
    start_date = pd.to_datetime(request.form['start_date'])
    end_date = pd.to_datetime(request.form['end_date'])
    item = int(request.form['item'])
    store = int(request.form['store'])
    frequency = request.form['frequency']
    
    # Filter data based on user input
    filtered_data = final_df[(final_df['date'] >= start_date) & (final_df['date'] <= end_date) & (final_df['item'] == item) & (final_df['store'] == store)]
    # Extract month from the 'date' column
    filtered_data['month'] = filtered_data['date'].dt.month
    # Extract year from the 'date' column
    filtered_data['year'] = filtered_data['date'].dt.year
    # Extract week number from the 'date' column
    filtered_data['week'] = filtered_data['date'].dt.isocalendar().week

    chart_data = filtered_data.groupby(['date']).size().reset_index(name='count')
    
    # Create pie chart and bar graph based on frequency
    if frequency == 'weekly':
        chart_data = filtered_data.groupby(['week'])['sales'].sum().reset_index()
    elif frequency == 'monthly':
        chart_data = filtered_data.groupby(['month'])['sales'].sum().reset_index()
    elif frequency == 'yearly':
        chart_data = filtered_data.groupby(['year'])['sales'].sum().reset_index()
    else:
        chart_data = filtered_data.groupby(['date'])['sales'].sum().reset_index()
    
    # Create Pie Chart
    pie_chart = px.pie(chart_data, values='sales', names=chart_data.columns[0], title=f'Sales Distribution - {frequency} ({start_date} to {end_date})')

    # Create Bar Graph
    bar_chart = px.bar(chart_data, x=chart_data.columns[0], y='sales', title=f'Sales Trends - {frequency} ({start_date} to {end_date})')
   
    # Create time series line chart
    time_series_chart = px.line(chart_data, x=chart_data.columns[0], y='sales', title=f'Sales Trends - {frequency} ({start_date} to {end_date})')

    # Convert the plot to HTML
    time_series_chart_html = time_series_chart.to_html(full_html=False)
    pie_chart_html = pie_chart.to_html(full_html=False)
    bar_chart_html = bar_chart.to_html(full_html=False)

    return render_template('forecast.html', pie_chart=pie_chart_html, bar_chart=bar_chart_html, time_series_chart=time_series_chart_html)

if __name__ == '__main__':
    app.run(debug=True)