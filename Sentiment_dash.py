# Importing the libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
import webbrowser
import dash.dependencies 
import dash_bootstrap_components as dbc
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

# Declaring Global variables
app=dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"
global data

# Defining My Functions
def load_model():
    global df
    global df2
    df=pd.read_csv('balanced_reviews.csv')
   
    
    df2=pd.read_csv('etsy_positivity.csv')
    
    
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)
    
    
    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
    
def check_review(review):
    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([review]))

    return pickle_model.predict(vectorised_review)

@app.callback(
    dash.dependencies.Output('alert2', 'children'),
    [
    dash.dependencies.Input('button_review', 'n_clicks')
    ]
    ,
    [
    dash.dependencies.State('textarea_review', 'value')
    ]
    
    )
def update_app_ui(n_clicks,textarea_value):
    print("Data Type  = ", str(type(n_clicks)))
    print("Value      = ", str(n_clicks)) 
    
    print("Data Type  = ", str(type(textarea_value)))
    print("Value      = ", str(textarea_value))

    if(n_clicks > 0):
        
        response = check_review(textarea_value)
    
        if (response[0] == 0 ):
            
            
            result =  dbc.Alert(
            "NEGATIVE",
            id="negative-alert",
            is_open=True,
            duration=8000,
            color="#DC143C"),
            
            
        elif (response[0] == 1 ):
           result =  dbc.Alert(
            "POSITIVE",
            id="positive-alert",
            is_open=True,
            duration=6000,
            color="	#00FFFF"  ), 
            
            
        else:
            result = 'Unknown'
    
         

    else:
        return ""
    
    return result
    
@app.callback(
    dash.dependencies.Output('alert1', 'children'),
    [
    dash.dependencies.Input('reviewer', 'value')
    ],
    
    
    )
def dropdown_review(value):
    print("Data Type  = ", str(type(value)))
    print("Value      = ", str(value)) 
    
   
        
    response = check_review(value)
    
    if (response[0] == 0 ):
       result =  dbc.Alert(
            "NEGATIVE",
            id="negative-alert",
            is_open=True,
            duration=6000,
            color="#DC143C"
            
        ),
    elif (response[0] == 1 ):
        result =  dbc.Alert(
            "POSITIVE",
            id="positive-alert",
            is_open=True,
            duration=6000,
            color="	#00FFFF"
        ),
    else:
        result = 'Unknown'
    
    return result  


@app.callback(
    dash.dependencies.Output('graphi', 'children'),
    
    
     [
    dash.dependencies.Input('subtab1', 'value'),
    dash.dependencies.Input('amazon', 'value'),
    dash.dependencies.Input('etsy', 'value'),
    ],    
 
    )

def graphs(s1,amazon_tab,etsy_tab):
    if(s1=='tab-1'):
        
        from plotly.subplots import make_subplots
        df.dropna(inplace=True)
        a=df['overall'].value_counts()
        x=a.to_dict()
        d=[[x[5],x[4]],[x[1],x[2]],[x[3],0]] 
        sas=np.array(d)

        labels=['Positive','Negative','Neutral']
        labels_2=[5,4,1,2,3,0]

        # graphs
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=sas.sum(axis=1),hole=.4),
              1, 1)
        fig.add_trace(go.Pie(labels=labels_2, values=sas.flatten(), name="Rating",hole=.7),
              1, 2)

        fig.update_traces(hoverinfo="label+percent+name+value")
        fig.update_layout(title="AMAZON REVIEW STATS",title_x=0.5,paper_bgcolor=" #FEF9E7 ")
       
        result=dcc.Graph(id='graph-object',figure=fig)

        return result
    
    elif(s1=='tab-2'):
        a1=df2['Positivity'].value_counts()
        x=a1.to_dict()

        l2=['Negative','Positive']
        v2=[x[0],x[1]]
        fig2=go.Figure(go.Pie(labels=l2,values=v2))
        fig2.update_layout(title="ETSY REVIEW STATS",title_x=0.5,paper_bgcolor=" #FEF9E7 ")
        result=dcc.Graph(id='graph-object-2',figure=fig2)
        
        return result
  
    
@app.callback(
     dash.dependencies.Output('drop-down','children'),
    [
    dash.dependencies.Input('subtab1', 'value'),
    dash.dependencies.Input('amazon', 'value'),
    dash.dependencies.Input('etsy', 'value'),
    ],     
    )
def tabs(s1,amazon_tab,etsy_tab):
    #global data
    global df
    global df2
    print("Tab value = " ,s1)
    print("Subtab1 value = " ,amazon_tab)
    print("Subtab2 value = " ,etsy_tab)
    
    if(s1=='tab-1'):
        result=dcc.Dropdown(id='reviewer',options=[{'label': i, 'value': i} for i in df['reviewText'].sample(30).values]

                     ,placeholder='Reviews', optionHeight=50),
       
        return result
    
    elif(s1=='tab-2'):
        result=dcc.Dropdown(id='reviewer',options=[{'label': i, 'value': i} for i in df2['Reviews'].sample(30).values]

                     ,placeholder='Reviews', optionHeight=50),
        
    
        return result
    

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')    
    

def create_app_ui():
    global data
    main_layout=html.Div(
    
    dbc.Jumbotron(
        
        [
        html.H1(id='Main_Title',children='Sentiment Analysis With Insights',style={'color': 'black','textAlign': 'center'}),
        
        dcc.Tabs(id="subtab1",value="tab-1",children=[
                    dcc.Tab(label="Amazon Reviews",id="amazon",value="tab-1"),
                    dcc.Tab(label="Etsy Reviews",id="etsy",value="tab-2")
                    ],
             colors={
        "border": "white",
        "primary": " #5499C7 ",
        "background": "    #D6EAF8   "
              }
            
            ),
       
        html.Div(id='graphi',children=[
            
        ]),
        html.Hr(),
       
       
        html.Div(
            children=[
            html.H5(children='Select Reviews From Below Dropdown',style={'textAlign':'center'}),
            html.Div(id='drop-down',children=[
                
                dcc.Dropdown(id='reviewer',placeholder='Reviews', optionHeight=50),
     
                ]
            
                     ),
      
        html.Br(),
        
        html.Div(id='alert1',children=[
            dbc.Alert(
                html.P(
            "Just select the Review from above Dropdown To Check It "),
                
                )
            
            ]),
      
        
        ]
        
        ),
      
        html.Br(),
        html.Hr(),
        html.H5(children='Enter Your Review Below',style={'textAlign':'center'}),

        
        dcc.Textarea(id='textarea_review',
                     placeholder='Enter Review Here',
                     style={'width':'100%','height':100}),
        
        html.Br(),
        
        dbc.Button(id='button_review',
                   children='Find Review',
                   style={'width':'100%'},
                   color='dark'),
        
        html.Br(),
        
          html.Div(id='alert2',children=[
            dbc.Alert(
                html.P(
            "Enter Your Review in The Above TextArea "),
                
                )]),     
    
  
        ]
        ),
    )
    return main_layout


   
#Main Function    
def main():
    global df
    global project_name
    print("Start of the Project")
    load_model()
    open_browser()

    
    
    project_name='Sentiment Analysis with Insights'
    print(project_name)
    app.title=project_name
    app.layout=create_app_ui()
    
    print(df.sample(5))
    
    #Blocking Statement
    app.run_server()

    
    print("End of the Project")
    df=None
    project_name=None
    
   
#Calling Main Function
if __name__=='__main__' :
    main()

    
    
 