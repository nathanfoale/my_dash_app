
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from flask import jsonify 

import dash_bootstrap_components as dbc
import base64
from flask import request
import os

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path to the Excel file (assuming it's in the same directory as your Python script)
file_path = os.path.join(BASE_DIR, 'Collingwood_Home_Games_Formatted.xlsx')

# now you can read the file as pandas dataframe
df = pd.read_excel(file_path, engine='openpyxl')



# Now, you can define routes using the Flask app
# Now, you can add endpoints to the Flask server for handling POST and GET requests.
@server.route('/your-endpoint', methods=['POST', 'GET'])
def handle_requests():
    if request.method == 'POST':
        # handle POST request here
        try:
            data = request.json
            # process your data, e.g., log it, store it, or trigger some other actions
            response = {"message": "Success, data received!"}  # customize as needed
            return jsonify(response)

        except Exception as e:
            response = {"error": "An error occurred: " + str(e)}
            return jsonify(response), 400

    elif request.method == 'GET':
        # handle GET request here
        try:
            # perhaps you're retrieving and sending back some data in this GET request
            data_to_return = {"key": "This is a response from a GET request"}
            return jsonify(data_to_return)

        except Exception as e:
            response = {"error": "An error occurred: " + str(e)}
            return jsonify(response), 400

    else:
        # Method not accounted for, return an error message
        return "Invalid request method", 405


#app = Dash(__name__)

# Load data
file_path = "Collingwood_Home_Games_Formatted.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df = df.sort_values('Year')

# column 'PrevYearLadderPosition'
df['PrevYearLadderPosition'] = df['Ladder Position'].shift(1)
df = df.dropna(subset=['PrevYearLadderPosition', 'Actual Crowd'])

# calculate average attendance and ending ladder position for each year
grouped_data = df.groupby('Year').agg({
    'Actual Crowd':'mean', 
    'Ladder Position':'last',
    'PrevYearLadderPosition': 'first'
}).reset_index()

grouped_data.loc[grouped_data['Year'] == 2022, 'Ladder Position'] = 4
grouped_data.loc[grouped_data['Year'] == 2023, 'Ladder Position'] = 1
grouped_data.loc[grouped_data['Year'] == 2018, 'Ladder Position'] = 3
grouped_data.loc[grouped_data['Year'] == 2016, 'Ladder Position'] = 12
grouped_data.loc[grouped_data['Year'] == 2014, 'Ladder Position'] = 11
grouped_data.loc[grouped_data['Year'] == 2011, 'Ladder Position'] = 1

X = grouped_data[['PrevYearLadderPosition']]
y = grouped_data['Actual Crowd']



model = LinearRegression()
model.fit(X, y)
grouped_data['Predicted'] = model.predict(X)

# create interactive plots using grouped_data
attendance_fig = px.scatter(grouped_data, x='Year', y='Actual Crowd',
                            title='Impact of Ladder Position on Attendance',
                            labels={'Year': 'Year', 'Actual Crowd': 'Average Attendance'},
                            size_max=15)

attendance_fig.update_layout(showlegend=True, template="simple_white")

r_squared = 0.704

ladder_fig = go.Figure()
ladder_fig.add_trace(
    go.Scatter(x=grouped_data['PrevYearLadderPosition'], y=grouped_data['Actual Crowd'],
               mode='markers', name='Actual Data',
               marker=dict(color='black'))  
)
ladder_fig.add_trace(
    go.Scatter(x=grouped_data['PrevYearLadderPosition'], y=grouped_data['Predicted'],
               mode='lines', name='Regression Line',
               line=dict(color='white'))  
)

ladder_fig.add_annotation(
    text=f'R<sup>2</sup> = {r_squared:.2f}',
    x=0.95,
    y=0.95,
    xref='paper',
    yref='paper',
    showarrow=False,
    font=dict(size=15, color="black")
)

ladder_fig.update_layout(
    title='OLS Regression Correlating Previous Season Ladder Finish and Attendance',
    xaxis_title='Previous Year Ladder Position',
    yaxis_title='Average Attendance',
    
    template="simple_white"
)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=grouped_data['Year'], y=grouped_data['Actual Crowd'], mode='lines+markers', name='Average Attendance', line=dict(color='white')),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=grouped_data['Year'], y=grouped_data['Ladder Position'], mode='lines+markers',
               name='Ending Ladder Position', line=dict(dash='dash', color='black')),
    secondary_y=True,
)
fig.update_layout(
    title='Average Attendance vs. Ending Ladder Position',
    xaxis_title='Year',
    yaxis_title='Average Attendance',
    yaxis2=dict(title='Previous Season Ladder Position', overlaying='y', side='right')
)

interstate_teams = {
    'West Coast': 'White',
    'Fremantle': 'White',
    'Adelaide': 'Black',
    'Port Adelaide': 'Black',
    'GWS': 'Black',
    'Sydney': 'Black',
    'Brisbane': 'Black',
    'Gold Coast': 'White'
}

# Filter the dataframe for Collingwood's home games against interstate teams
filtered_df = df[df['Team'] == 'Collingwood']
filtered_df = filtered_df[filtered_df['Opposition'].isin(interstate_teams)]

# Group by the opposition and calculate the average crowd
avg_crowds = filtered_df.groupby('Opposition')['Actual Crowd'].mean().sort_values(ascending=False).reset_index()

# Create an interactive bar chart with custom colors
fig1 = px.bar(
    avg_crowds,
    x='Opposition',
    y='Actual Crowd',
    title="Average Attendance for Collingwood's Home Games Against Interstate Teams",
    color='Opposition',  # Use 'Opposition' as the color mapping variable
    color_discrete_map=interstate_teams  # Assign colors based on the dictionary
)



# Box and whisker plots

df['PrevGameResult'] = df['Win/Loss'].shift(1)
df['NextGameAttendance'] = df['Actual Crowd'].shift(-1)
# # Data preparation
win_data = df[df['PrevGameResult'] == 'Win']['NextGameAttendance']
loss_data = df[df['PrevGameResult'] == 'Loss']['NextGameAttendance']

# # Create the box whisker figure
box_whisker_fig = go.Figure()

# # Adding boxes for 'Win' and 'Loss'
box_whisker_fig.add_trace(go.Box(y=win_data, name="Win", marker_color="black"))
box_whisker_fig.add_trace(go.Box(y=loss_data, name="Loss", marker_color="white"))

# # Layout adjustments
box_whisker_fig.update_layout(
     title="Box and Whisker Plot for Next Game Attendance based on Previous Game's Result",
     yaxis_title="Next Game Attendance"
 )



def update_graph_styles(figure):
    figure.update_layout(
        paper_bgcolor='palegoldenrod',
        plot_bgcolor='palegoldenrod',
        font=dict(color='black')
    )

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Filter for games on Saturday and Sunday
weekend_games_df = df[df['Day'].isin(['Saturday', 'Sunday'])]

# Aggregate data by 'Start Time' and 'Day' to get average attendance
agg_attendance = weekend_games_df.groupby(['Day', 'Start Time'])['Actual Crowd'].mean().reset_index()

# Compute the overall average attendance
overall_avg_attendance = df['Actual Crowd'].mean()

# Bar chart for average attendance based on match timing and day
fig3 = px.bar(agg_attendance, x='Start Time', y='Actual Crowd', color='Start Time',
             facet_col='Day',
             title='Average Attendance based on Match Timing and Day for Collingwood Home Games',
             labels={'Start Time': 'Match Timing', 'Actual Crowd': 'Average Attendance'})

# Add a horizontal dotted line for the overall average attendance in each facet
for i in range(1, len(fig3.layout.annotations) + 1):
    fig3.add_shape(
        go.layout.Shape(
            type="line",
            y0=overall_avg_attendance,
            y1=overall_avg_attendance,
            xref=f'x{i}',
            yref=f'y{i}',
            line=dict(dash="dot", color="red")
        )
    )

# Adjust bargap to remove gaps within bars of each day
fig3.update_layout(bargap=0)
fig3.update_xaxes(tickangle=-45)

# Update x-axis for each subplot based on available data points
days = ['Saturday', 'Sunday']
for index, day in enumerate(days, start=1):
    unique_times = agg_attendance[agg_attendance['Day'] == day]['Start Time'].unique()
    fig3.update_xaxes(tickvals=unique_times, ticktext=unique_times, col=index, categoryorder="array", categoryarray=unique_times)

graphs = [attendance_fig, ladder_fig, fig, fig1, box_whisker_fig, fig3]
for graph in graphs:
    update_graph_styles(graph)


dropdown_style = {'backgroundColor': 'palegoldenrod', 'color': 'black'}  





app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Collingwood Football Club Home Attendance Analysis Dashboard (2011-2023)', style={
            'color': 'Black',
            'font-size': '2.5em',
            'padding': '10px 20px',
            'text-align': 'center',
        }), className="mb-4 text-center mx-auto"),
    ]),
    dbc.Row([
       dbc.Col(
    #html.Img(src='/Users/nathanfoale/assets/penguins.png')
),
    ]),
    # Row for the first two graphs
    dbc.Row([

        dbc.Col(dcc.Graph(id='attendance-vs-ending-ladder-graph',
                          figure=fig.update_layout(
                              legend=dict(font=dict(size=10)),
                              xaxis=dict(title_font=dict(size=14), gridcolor='rgba(0,0,0,0)'),
                              yaxis=dict(title_font=dict(size=14), gridcolor='rgba(0,0,0,0)'),
                              title=dict(text='Average Attendance vs. Previous Seasons Ladder Position', font=dict(size=16))
                          )),
                width=6, className="mb-4"),
        
        dbc.Col(dcc.Graph(id='ladder-graph', figure=ladder_fig), width=6, className="mb-4"),

    ]),
    # Row for the bottom two graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id='attendance-graph', figure=fig1), width=6, className="mb-4"),
        dbc.Col(dcc.Graph(id='box-whisker-graph', figure=box_whisker_fig), width=6, className="mb-4")
    ]),
    dbc.Row([
        # Column for dropdown
        dbc.Col([
            html.Div(  
                dcc.Dropdown(
                        id='day-dropdown',
                        options=[
                            {'label': 'Saturday', 'value': 'Saturday'},
                            {'label': 'Sunday', 'value': 'Sunday'}
                        ],
                        placeholder='Select day',  
                        clearable=False,  
   
                ),
                style={
                    'width': '7%',  
                    'padding': '10px',  
                    'margin-left': '45%',  
                    'margin-right': '25%',  
                }
            ),
        ], width=12),  
    ]),

    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='attendance-time-day-graph')
        ], width=12)  
    ]),
 
], fluid=True, class_name="container-gradient", style={"backgroundColor": "palegoldenrod"})

@app.callback(
    Output('attendance-time-day-graph', 'figure'),
    [Input('day-dropdown', 'value')]
)
def update_graph(selected_day):
    # filter dataframe based on day
    filtered_df = df[df['Day'] == selected_day]

   
    agg_attendance = filtered_df.groupby('Start Time')['Actual Crowd'].mean().reset_index()

  
    overall_avg_attendance = filtered_df['Actual Crowd'].mean()


    fig3 = go.Figure()

  
    fig3.add_trace(
        go.Bar(
            x=agg_attendance['Start Time'],
            y=agg_attendance['Actual Crowd'],
            marker_color=["black" if i % 2 == 0 else "white" for i in range(len(agg_attendance))],  
            name='Average Attendance'
        )
    )

    
    fig3.add_shape(
        type="line",
        x0=agg_attendance['Start Time'].iloc[0],  # starting from the first Start Time
        x1=agg_attendance['Start Time'].iloc[-1],  # ending at the last Start Time
        y0=overall_avg_attendance,
        y1=overall_avg_attendance,
        line=dict(
            color="Red",
            width=4,
            dash="dashdot",
        )
    )
    

  
    fig3.update_layout(
        title='Average Attendance based on Match Timing for {}'.format(selected_day),
        xaxis_title='Start Time',
        yaxis_title='Average Attendance',
        showlegend=True,
        
        paper_bgcolor='palegoldenrod',
        plot_bgcolor='palegoldenrod',
        font=dict(color='black')
    )

    return fig3
# Adjust bargap to remove gaps within bars of each day
fig3.update_layout(bargap=0)
fig3.update_xaxes(tickangle=-45)

# Update x-axis for each subplot based on available data points
days = ['Saturday', 'Sunday']
for index, day in enumerate(days, start=1):
    unique_times = agg_attendance[agg_attendance['Day'] == day]['Start Time'].unique()
    fig3.update_xaxes(tickvals=unique_times, ticktext=unique_times, col=index, categoryorder="array", categoryarray=unique_times)


if __name__ == '__main__':
    app.run_server(debug=True)
    
# @app.callback(
#     [Output('venue-analysis', 'figure'),
#      Output('opponent-analysis', 'figure')],
#     Input('club-dropdown', 'value')
# )
# def update_plots(selected_club):
#     # Ensure data filtering is correct
#     filtered_df = df2[df2['Team'] == selected_club]

#     # Venue Analysis
#     venue_group = filtered_df.groupby('Venue').agg({'Final Score': 'mean'}).reset_index()
#     venue_fig = px.bar(venue_group, x='Venue', y='Final Score', 
#                        title=f"Average Scores For at Different Venues for {selected_club}")

#     # IMPORTANT: Update traces here to change the bar color before setting the layout
#     venue_fig.update_traces(marker_color='black')  # Change the bar color to black

#     venue_fig.update_layout(
#         paper_bgcolor='palegoldenrod',
#         plot_bgcolor='palegoldenrod',
#         font=dict(color='black')
#     )
#     fig3.update_layout(
#         paper_bgcolor='palegoldenrod',
#         plot_bgcolor='palegoldenrod',
#         font=dict(color='black')

#     )

#     # Opponent Analysis
#     opponent_group = filtered_df.groupby('Opposition').agg({'Final Score': 'mean', 'Actual Crowd': 'mean'}).reset_index()
#     opponent_fig = px.bar(opponent_group, x='Opposition', y='Final Score', 
#                           title=f"Average Scores Versus Different Opponents for {selected_club}")

#     # IMPORTANT: Update traces here to change the bar color before setting the layout
#     opponent_fig.update_traces(marker_color='white')  # Change the bar color to black

#     opponent_fig.update_layout(
#         paper_bgcolor='palegoldenrod',
#         plot_bgcolor='palegoldenrod',
#         font=dict(color='black')
#     )

#     return venue_fig, opponent_fig, fig3



    
# app.layout = html.Div([
#     html.H1('Collingwood Attendance Analysis Dashboard'),
#     html.Div('Dash: A web application framework for Python.'),
    
#     # Row for the first two graphs
#     html.Div([
#         # Div for the ladder graph
#         html.Div([
#             dcc.Graph(id='ladder-graph', figure=ladder_fig),
#         ], style={'width': '48%', 'display': 'inline-block'}),

#         # Div for the attendance vs ending ladder graph
#         html.Div([
#             dcc.Graph(id='attendance-vs-ending-ladder-graph', figure=fig),
#         ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
#     ], style={'clear': 'both'}),
    
#     # Row for the bottom two graphs
#     html.Div([
#         # Div for the attendance graph
#         html.Div([
#             dcc.Graph(id='attendance-graph', figure=fig1),
#         ], style={'width': '48%', 'display': 'inline-block'}), 

#         # Div for the box-whisker graph
#         html.Div([
#             dcc.Graph(id='box-whisker-graph', figure=box_whisker_fig),
#         ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
#     ], style={'clear': 'both'}),
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
