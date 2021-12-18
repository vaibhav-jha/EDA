# Installations required  (IMPORTANT)
# pip install numerize
# conda install -c conda-forge folium
# make sure the file path also include the ./assets folder that contains additional .css and images

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt



import dash
from dash import dcc
from jupyter_dash import JupyterDash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from numerize import numerize

import folium
from folium import plugins


## Data Loading and Manipulation

data = pd.read_csv('US_Accidents_Dec20_updated.csv')
# dropping Number (too many null values)
data.drop(columns=['Number'], inplace=True)
# dropping Description because we don't use it
data.drop(columns=['Description'], inplace=True)


# Adding a Date column and a Weekday column
data['Date'] = pd.to_datetime(data['Start_Time']).dt.date
weekday_dict = dict(zip(range(0,8), ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']))
data['Weekday'] = pd.to_datetime(data.Date).dt.weekday.map(lambda x: weekday_dict[x])
data['Year'] = pd.to_datetime(data.Date).dt.year
data['Month'] = pd.to_datetime(data.Date).dt.month

# Adding Time of day
def time_of_day(t):
    if t < 2:
        t = 24
    slot = int((t-2)/4)
    if slot == 5:
        return '22 to 2'
    return f'{slot*4+2} to {slot*4+6}'

data['Time_of_Day'] = pd.to_datetime(data['Start_Time']).dt.hour.map(lambda x: time_of_day(x))
data.Time_of_Day = pd.Categorical(data.Time_of_Day, 
                                categories=sorted(data.Time_of_Day.unique(), key=lambda x: int(x[:2].strip())))


data['Traffic_Affected_Hrs'] = ((pd.to_datetime(data.End_Time) - pd.to_datetime(data.Start_Time))\
                                .dt.total_seconds()/3600).map(lambda x: round(x,2))

data['Junction%'] = data.groupby('State').Junction.sum() * 100 / data.groupby('State').Junction.count()


## -----------------------------------------------------------------------------------------------------
import dash
from dash import dcc
from jupyter_dash import JupyterDash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from numerize import numerize

import folium
from folium import plugins


# app = JupyterDash(__name__, external_stylesheets=[dbc.themes.LUMEN])
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])

data_grp_state = data.groupby(['State', 'Severity']).count()

fig1 = go.Figure(data=go.Choropleth(
    locations=data_grp_state.index.levels[0], # Spatial coordinates
    z = data_grp_state.groupby(level=0).sum().ID, # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'reds',
    colorbar_title = "Number of Accidents",
))

fig1.update_layout(
    title_text = 'Select a State from the map below:',
    geo_scope='usa', # limite map scope to USA
    dragmode=False,
    height=500,
    margin=dict(l=0, r=0, t=80, b=10),
    paper_bgcolor='rgba(0,0,0,0)',
    geo=dict(bgcolor= 'rgba(0,0,0,0)'),
)

fig1.data[0].colorbar.x=-0.02

# dimensions = [dict(values=data[label], label=val) \
#                  for label, val in {'Sunrise_Sunset': 'Time_of_Day'\
#                                , 'Junction': 'Junction'\
#                                , 'Severity': 'Severity'}.items()]


# fig2 = go.Figure(go.Parcats(
#         dimensions=dimensions,
#         ))
# fig2 = px.parallel_categories(data, dimensions=['Sunrise_Sunset',\
#                                                         'Weekday',
#                                                         'Junction',]
#                             ,labels = {'Sunrise_Sunset': 'Time_of_Day'\
#                                , 'Junction': 'Junction'\
#                                , 'Weekday': 'Day_of_Week'}
#                             , color='Severity', color_continuous_scale='YlGn')

# fig2.update_layout(
#     paper_bgcolor = 'rgba(0,0,0,0)',
#     plot_bgcolor = 'rgba(0,0,0,0)',
# )



## Hann
dropdown_top_10 = dcc.Dropdown(
        id='top-10-picker',
        options=[
            {'label': 'State', 'value': 'State'},
            {'label': 'City', 'value': 'City'},
            {'label': 'Street', 'value': 'Street'}
        ],
        value='State',
        style={'margin-bottom': '3px', 'background': 'rgba(0,0,0,0)'},
    )

_kpi1 = numerize.numerize(round(data.groupby(['Year', 'Month']).count().State.mean(), 2))
usa_jn_perc = data.Junction.sum() * 100 / data.Junction.count()
_kpi2 = str(round(usa_jn_perc, 2))+'%'
_kpi3 = str(round(data['Distance(mi)'].mean(), 2))+'mi'

#----------------------------------------------------------------------------------------------------------------
## Utility Methods

def generate_summary_table(df):
    
    ret_df = df.describe().loc[['mean', '50%', 'max', 'min'], ['Visibility(mi)', 'Humidity(%)', 'Temperature(F)', 'Wind_Speed(mph)', 'Pressure(in)','Distance(mi)', 'Traffic_Affected_Hrs']]
    ret_df.index = ['Average', 'Median', 'Max', 'Min']
    
    return ret_df
        
    
def quartile3(x):
    return x.quantile(0.75)
def quartile1(x):
    return x.quantile(0.25)

#----------------------------------------------------------------------------------------------------------------


## APP LAYOUT

app.layout = html.Div([
   
    ## Banner
    html.Img(src='/assets/dso545_banner_5.png'
                , style={'width': '100%'}
            ),
    
    ## Title
    html.Center([
        
    
#         html.H1('US Car Accidents Analyses', id='heading'),
        dcc.Loading(
            id="loading-subheading",
            type="default",
            color = '#fc9272',
            children= [ 
                html.P('State : All', id='subheading', style={'color': 'gray'})
            ]

        ),
        dcc.Link('Show All States', href='/', id='showall', refresh=True),
    
    ]),
    

    
    ## MAP and KPIs
    html.Div([
        
        ## Map
        html.Div([
            dcc.Loading(
                id="loading-map",
                type="graph",
                color = '#FFFFFF',
                children= [dcc.Graph(
                            id='map',
                            figure=fig1,   
                        ),
                           ## Button
                           html.Div([
                               dbc.Button("Heat Map", color="info", id="open-modal", 
                                         style={'margin': 'auto', 'display': 'block'}),
                           ], ),
                            
                ])
            
        ], style={'width': '50%', 'overflow': 'hidden', 'cursor': 'pointer', 'display': 'inline-block'
                 , 'vertical-align': 'top'}
        ),        
        
        
        ## Table
        html.Div([
            
            dcc.Loading(
                id='table-load',
                type='circle',
                color = '#fc9272',
                children=[html.Div(id='summary-table-div',),]
            ),
            
            
                ## KPIs
                html.Div([
                    dbc.Card([
                        dbc.CardHeader("Avg Monthly Accidents", class_name='kpi-header'),
                        dbc.CardBody(
                            [
                                dcc.Loading(
                                    id='kpi1-load',
                                    type='dot',
                                    color = '#FFFFFF',
                                    children=[html.H4(_kpi1, className="card-title", id='kpi1'),]),
                                
#                                 html.P(
#                                     "Average Monthly Accidents",
#                                     className="card-text",
#                                 ),
                            ]
                        ),
                    ], color="danger", inverse=True
                        , style={'width': '25%', 'padding-bottom': '25%', 'height': '0'
                                 ,'display': 'inline-block', 'border-radius':'50%'
                                ,'text-align': 'center', 'vertical-align': 'top'},),

                    dbc.Card([
                        dbc.CardHeader("% Accidents at Junctions", class_name='kpi-header'),
                        dbc.CardBody(
                            [
                                dcc.Loading(
                                    id='kpi2-load',
                                    type='dot',
                                    color = '#FFFFFF',
                                    children=[html.H4(_kpi2, className="card-title",  id='kpi2'),]),
#                                 html.P(
#                                     "blah blah",
#                                     className="card-text",
#                                 ),
                            ]
                        ),
                    ], color="warning", inverse=True
                        , style={'width': '25%', 'padding-bottom': '25%', 'height': '0'
                                 ,'display': 'inline-block', 'border-radius':'50%'
                                ,'text-align': 'center', 'vertical-align': 'top'},),

                    dbc.Card([
                        dbc.CardHeader("Average Affected Distance", class_name='kpi-header'),
                        dbc.CardBody(
                            [
                                dcc.Loading(
                                    id='kpi3-load',
                                    type='dot',
                                    color = '#FFFFFF',
                                    children=[html.H4(_kpi3, className="card-title", id='kpi3'),]),
                                
#                                 html.P(
#                                     "blah blah blah",
#                                     className="card-text",
#                                 ),
                            ]
                        ),
                    ], color="success", inverse=True
                        , style={'width': '25%', 'padding-bottom': '25%', 'height': '0'
                                 ,'display': 'inline-block', 'border-radius':'50%'
                                ,'text-align': 'center', 'vertical-align': 'top'}, ),
                ],),
        
        ],style={'width': '47%', 'display': 'inline-block', 'vertical-align': 'bottom'}, id='summary-and-kpis'),
        
    ], style={'margin': '10px'}),
    
    
    html.Div([
        ## PARCAT
        html.Div([
            dcc.Loading(
                id="loading-parcat",
                type="default",
                color = '#fc9272',
                children=dcc.Graph(id='example-graph-3')
            )
        ] , style={'width': '48%', 'overflow': 'hidden', 'display': 'inline-block', 'vertical-align': 'top'}, className='han_div'),


        ## Hann

        html.Div([
            dropdown_top_10,
            dcc.Loading(
                id='loading-bar',
                type='default',
                color="#fc9272",
                children= [dcc.Graph(
                    id = 'top-10-graph',
                    )],
                style={'margin': '6px'}
            )
        ], 
        style={'width': '48%', 'display': 'inline-block'}, className='han_div'),
    ]),
    
    
    
    ## Sursi
    
     html.Div([
        html.H3("External Conditions", style={'text-align': 'center'}), 
         
         dbc.RadioItems(
             id='weather-param',
             options=[{'value': x, 'label': x} for x in ['Temperature(F)','Precipitation(in)','Humidity(%)','Wind_Speed(mph)','Visibility(mi)']],
             value='Humidity(%)', 
             labelCheckedClassName="text-success",
             inputCheckedClassName="border border-success bg-success",
             inline=True,
             style={'text-align': 'center'}
         ),
         
         dcc.Loading(
             id='loading-box-pie',
             type='default',
             color="#fc9272",
             children=[
                 dcc.Graph(id="box-plot", style={'display': 'inline-block', 'width': '49%'}),
                 dcc.Graph(id="pie-chart", style={'display': 'inline-block', 'width': '49%'}),
             ])
         
        ], className='han_div'),

    
    ## Hema
#     html.Div([
#         dcc.Loading(
#             id='heatmap_load',
#             type='circle',
#             children=[
#                 html.Iframe(id='the_heat_map', width='100%', height='600'),
#             ]
#         )
#     ], style={'width': '80%', 'margin': 'auto',}),
    
    
    ## Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Heatmap", id='heatmap-title'), close_button=True),
                dbc.ModalBody(
                    html.Div([
                        dcc.Loading(
                            id='heatmap_load',
                            type='circle',
                            color="#fc9272",
                            children=[
                                html.Div([
                                    html.Iframe(id='the_heat_map', width='96%', height='400px'
                                                , style={'display': 'inline-block', 'border-color':'black'
                                                , 'border-width':'5px'
                                                , 'border-style': 'solid'
                                                , 'margin': '2%'}),
#                                     html.Div([
#                                         ## TODO: Add Stats for the state in Modal
#                                         html.P("Some stats go here")
#                                     ], style={'width':'39%', 'height':'100%', 'display': 'inline-block', 'vertical-align':'top'})
#                                 ], style = {'height':'69%'}),
                            ]),
                                
#                                 html.Div([
#                                     html.P("More stats go here")
#                                 ], style = {'height':'30%'} )
                            ]
                        )
                    ], style={'width': '90%', 'margin': 'auto',}),
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-centered",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-centered",
            centered=True,
            is_open=False,
            size='xl',
        ),
    
])

## APP CALLBACKS
@app.callback(
    [Output('map', 'figure'),
     Output('subheading', 'children'),],
    [Input('map', 'clickData'),
    ])
def change_map(clickData):
    
#     ctx = dash.callback_context

#     if not ctx.triggered:
#         button_id = 'No clicks yet'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
#     if button_id == 'subheading':
#         return fig1, 'State : All'
    
    ## If we are here, the map was clicked on some State
    if clickData == None:
        raise PreventUpdate
    
    
    st = clickData['points'][0]['location']
    cl_scale = [[0, 'rgb(255,237,160)'], [1, 'rgb(252,146,114)']]
    z = [1 if i==st else 0 for i in data_grp_state.index.levels[0]]
    
    fig_r = go.Figure(data=go.Choropleth(
                            locations=data_grp_state.index.levels[0], # Spatial coordinates
                            z = z, # Data to be color-coded
                            locationmode = 'USA-states', # set of locations match entries in `locations`
                            colorscale = cl_scale,
                            colorbar_title = "Number of Accidents",
                            showscale = False,
                            )
                     )
    fig_r.update_layout(
            title_text = 'Select a State from the map below:',
            geo_scope='usa', # limite map scope to USA
            dragmode=False,
            height=500,
            margin=dict(l=0, r=0, t=0, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor= 'rgba(0,0,0,0)'),
            coloraxis_showscale=False,
        )
    return fig_r, f'State : {st}' if st else 'State : All'

@app.callback(
    Output('example-graph-3', 'figure'),
    [Input('map', 'clickData')])
def update_parcat(clickData):
    
    data_sub = data
    
    if clickData != None:
        st = clickData['points'][0]['location']
        data_sub = data[data['State'] == st]
    
    
        # Create dimensions
    day_night_dim = go.parcats.Dimension(values=data_sub.Sunrise_Sunset, label="Day or Night")

    junction_dim = go.parcats.Dimension(values=data_sub.Junction.astype(int), label="Junction"
                                        , categoryarray=[0,1]
                                        , ticktext=['No Junction', 'Junction']
                                       )

    dow_dim = go.parcats.Dimension(values=data_sub.Weekday, label="Day of Week",
                                     categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


    sev_labels = {
        1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme',
    }
    
    unique_sev = sorted(data_sub.Severity.unique())

    severity_dim = go.parcats.Dimension(
        values=data_sub.Severity, label="Severity"
        , categoryarray=unique_sev
        , ticktext=[sev_labels[i] for i in unique_sev]
    )

    # Create parcats trace
    color = data_sub.Severity;

    fig_r = go.Figure(data = [go.Parcats(dimensions=[day_night_dim, dow_dim, junction_dim, severity_dim],
            line={'color': color, 'colorscale': 'YlGn'},
            hoveron='color', hoverinfo='count+probability',
            labelfont={'size': 18, 'family': 'Times'},
            tickfont={'size': 16, 'family': 'Times'},
            arrangement='freeform')])

#     fig_r = px.parallel_categories(data_sub, dimensions=['Sunrise_Sunset',\
#                                                         'Weekday',
#                                                         'Junction',]
#                             ,labels = {'Sunrise_Sunset': 'Time_of_Day'\
#                                , 'Junction': 'Junction'\
#                                , 'Weekday': 'Day_of_Week'}
#                             , color='Severity', color_continuous_scale='YlGn',
#                                   )
    
#     fig_r.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)'
#     )
    return fig_r

## Hann
@app.callback(
    [Output('top-10-picker', 'options'),
     Output('top-10-picker', 'value'),],
    [Input('map', 'clickData'),])
def change_bar_dropdown(state):
    if state in [None, 'All']:
        raise PreventUpdate

    options=[
        {'label': 'City', 'value': 'City'},
        {'label': 'Street', 'value': 'Street'}
    ]
    
    return options, 'City'

@app.callback(
    Output('top-10-graph', 'figure'),
    [Input('map', 'clickData'),
     Input('top-10-picker', 'value'),
    ])
def update_bar_graph(state, top_10_picker):
    if state == None and top_10_picker == None:
        raise PreventUpdate

#     print(state, top_10_picker)
    if state not in [None, 'All']:
        top_10_picker_label = 'Cities' if top_10_picker == 'City' else top_10_picker+'s' 
        st = state['points'][0]['location']
        data_st = data[data['State'] == st]
        title = f'Top 10 Accident Prone {top_10_picker_label} in the state of {st} (2016-2020)'
        hover_data = []
    else:
        top_10_picker_label = 'Cities' if top_10_picker == 'City' else top_10_picker+'s'
        data_st = data
        title = f'Top 10 Accident Prone {top_10_picker_label} in the US (2016-2020)'
        hover_data = ['State']
    
    top_10_df = pd.DataFrame(data_st[top_10_picker].value_counts()).reset_index().rename(columns={'index':top_10_picker, top_10_picker:'Cases'}).head(10)

    fig = px.bar(x= top_10_df[top_10_picker], y= top_10_df['Cases']
                 , title= 'Top 10'
#                  , hover_data= top_10_df[hover_data]
                )
    fig.update_xaxes(title = top_10_picker)
    fig.update_yaxes(title = 'Cases')
    fig.update_traces(marker_color='rgb(250,218,94)')
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title = title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

### Table

@app.callback(
    [Output('summary-table-div', 'children'),],
    [Input('map', 'clickData'),],
)
def update_table(clickData):
    subsetData = data
    if clickData not in [None, 'All']:
        st = clickData['points'][0]['location']
        subsetData = data[data['State'] == st]
        
    summ_data = generate_summary_table(subsetData)
    summ_data = summ_data.applymap(lambda x: round(x, 2))
    summ_data = summ_data.T.reset_index()
    summ_data.columns = ['Summary',*summ_data.columns[1:].tolist()]
        
    summ = dbc.Table.from_dataframe(summ_data, striped=True, bordered=True, hover=True
                                    , size='sm'
                                    , color='success'
                                   , class_name='summary_table')
    
    return [summ]
    

    
### KPIs
@app.callback(
    [Output("kpi1", "children"),
     Output("kpi2", "children"),
     Output("kpi3", "children"),
    ],
    [Input("map", "clickData"),]
)
def update_kpis(clickData):
    if clickData == None:
        raise PreventUpdate
    
    st = 'All'
    subsetData = data
    if clickData not in [None, 'All']:
        st = clickData['points'][0]['location']
        subsetData = data[data['State'] == st]
    
    # KPI 1 - Avg Monthly Accidents
    kpi1 = round(subsetData.groupby(['Year', 'Month']).count().State.mean(), 2)
    
    #KPI 2 - Junction Accidents 
    state_level_jn_perc = data.groupby('State').Junction.sum() * 100 / data.groupby('State').Junction.count()
    usa_jn_perc = data.Junction.sum() * 100 / data.Junction.count()
    kpi2 = str(round(state_level_jn_perc[st] if st != 'All' else usa_jn_perc, 2))+'%'
    
    #KPI 3 - Average Distance Affected
    kpi3 = str(round(subsetData['Distance(mi)'].mean(), 2))+'mi'
    
    kpi1 = numerize.numerize(kpi1)
    
    return kpi1, kpi2, kpi3
    

### Sursi
@app.callback(
    [Output("box-plot", "figure"),
     Output("pie-chart", "figure"),],
    [Input("weather-param", "value"),
     Input("map", "clickData")],
    [State('weather-param', 'options'),]
)
def update_box_pie(y, clickData, options):
    if clickData == None and y == None:
        raise PreventUpdate
        
    subsetData = data
    if clickData not in [None, 'All']:
        st = clickData['points'][0]['location']
        subsetData = data[data['State'] == st]
    
    if y==None:
        y = 'Humidity(%)'
#     ## Zoom into include only few outliers
#     max_3q = subsetData.groupby('Severity').agg(quartile3)[y].max()
#     min_1q = subsetData.groupby('Severity').agg(quartile1)[y].min()
#     iqr = max_3q - min_1q
#     ymax = subsetData[y].quantile(0.75) + 1.5*iqr
#     ymin = subsetData[y].quantile(0.25) - 1.5*iqr
    
    
    fig2 = px.box(subsetData, x='Severity', y=y, template='plotly_white')
    fig2.update_traces(marker_color='rgb(252,146,114)')
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         yaxis=dict(range=[ymin, ymax]),
    )
    
    
#     vals = list(map(lambda x: x['value'], options))
    pie_data = pd.cut(subsetData[y], bins=5).value_counts().to_frame().reset_index()
#     labels = pie_data['index'].apply(lambda x: round(x.mid,2))
    labels = pie_data['index'].astype(str).tolist()

    vals = pie_data[y]
    
    pie = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=.3)])
    pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    pie.update_layout(
        title=f'Distribution of {y}',
    )
    pie.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=px.colors.sequential.YlGn[-5:], line=dict(color='#000000', width=2)))
    
    
    return fig2, pie

### HEATMAP

@app.callback([Output('the_heat_map', 'srcDoc')],
              [Input('map', 'clickData')])
def update_heatmap(clickData):
#     if clickData == None or 'location' not in clickData['points'][0]:
#         raise PreventUpdate
    geo_df_state = data
    zoom_start = 3
    
    if clickData != None and 'location' in clickData['points'][0]:
        st = clickData['points'][0]['location']
        geo_df_state = data[data['State'] == st]
        zoom_start = 6
    
    #Map out centerpoint to initiate the map
    m = folium.Map([(geo_df_state['Start_Lat'].min()+geo_df_state['Start_Lat'].max())/2, 
                                     (geo_df_state['Start_Lng'].min()+geo_df_state['Start_Lng'].max())/2],
#                                       width='60%', height='50%',
                                      zoom_start=zoom_start)

    # convert to (n, 2) nd-array format for heatmap
    all_points = geo_df_state[['Start_Lat', 'Start_Lng']].to_numpy()

    #plot heatmap
    m.add_child(plugins.HeatMap(all_points, radius=12))
    
    m.save('heatmap.html')
    return [open('heatmap.html', 'r').read()]

## MODAL 

@app.callback(
    Output("modal-centered", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-centered", "n_clicks")],
    [State("modal-centered", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

if __name__ == '__main__':
    app.run_server(debug=True)