
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import json

import plotly
import plotly.express as px
import plotly.graph_objects as go

import os
from os.path import join, dirname, realpath

import github

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/csv'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

app.config['TEMPLATES_AUTO_RELOAD'] = True

csv_list = []

# session.file

class session:
    fileSelected = 'Hipoteca.csv'
    url = ''
    fileEDA = 'melb_data.csv'
    filePCA = 'Hipoteca.csv'
    fileArboles = 'diabetes.csv'
    fileSegmentacion = 'diabetes.csv'
    fileSoporteVectorial = 'diabetes.csv'
    arbolesRes = {}

# -------- PAGINAS ---------------------------------------------------

# Root URL
@app.route('/')
def index():
    return render_template('index.html')#,  graphJSON=gm())

@app.route('/eda')
def eda():
    update_csv_dir()
    session.url = url_for('eda')
    return render_template('eda.html', csv_list=csv_list)

# Get the uploaded files for eda
@app.route("/eda", methods=['POST'])
def uploadFilesEDA():
    return uploadFiles()

@app.route('/pca')
def pca():
    update_csv_dir()
    session.url = url_for('pca')
    #df = pd.read_csv('https://raw.githubusercontent.com/PizzaDude007/MineriaDatos/main/Datos/Hipoteca.csv')
    fileName = request.args.get('fileData')
    
    if (session.fileSelected is not None and fileName is not None):
        session.fileSelected = fileName

    print('Internal: '+str(session.fileSelected))
    print('External: '+str(fileName))
    
    df = pd.read_csv('static/csv/'+session.fileSelected)

    return render_template('pca.html', table=df, pd=pd, nameData=session.fileSelected, csv_list=csv_list)

# Get the uploaded files for pca
@app.route("/pca", methods=['POST'])
def uploadFilesPCA():
    return uploadFiles()

@app.route('/arboles')
def arboles():
    update_csv_dir()
    session.url = url_for('arboles')
    fileName = request.args.get('fileName')
    x = request.args.getlist('valorX') # Variables Predictoras
    y = request.args.get('valorY') # Variable a Pronosticas

    df = pd.read_csv('static/csv/'+session.fileSelected)
    pronostico = ''

    print('Internal: '+str(session.fileSelected))
    print('External: '+str(fileName))

    if (x is not None and y is not None):
        print(str(x))
        #print(y)
        session.arbolesRes = train(df, x, y)
        #print("Variables Precitoras: ")
        #print(session.arbolesRes['X'])
        #print("Variable a Predecir: ")
        #print(session.arbolesRes['Y_Pronostico'])
        
        dictOpciones = {}
        for name in session.arbolesRes['X']:
            if(request.args.get(name) is not None or request.args.get(name) is FileNotFoundError):
                dictOpciones[name] = float(request.args.get(name))
                print(dictOpciones[name])
                pronostico = str(obtener_pronostico(dictOpciones, session.arbolesRes['Pronostico']))
    
    return render_template('arboles.html', table=df, nameData=session.fileSelected, res=session.arbolesRes, pronostico=pronostico, csv_list=csv_list)

# Get the uploaded files
@app.route("/arboles", methods=['POST'])
def uploadFilesArboles():
    return uploadFiles()

@app.route('/bosques')
def bosques():
    update_csv_dir()
    session.url = url_for('bosques')
    return render_template('bosques.html', csv_list=csv_list)

# Get the uploaded files
@app.route("/bosques", methods=['POST'])
def uploadFilesBosques():
    return uploadFiles()

# Segmentacion
@app.route('/segmentacion')
def segmentacion():
    update_csv_dir()
    session.url = url_for('segmentacion')
    return render_template('segmentacion.html', csv_list=csv_list)

# Get the uploaded files for segmentacion
@app.route("/segmentacion", methods=['POST'])
def uploadFilesSegmentacion():
    return uploadFiles()

# Soporte vectorial
@app.route('/soporte_vectorial')
def soporte_vectorial():
    update_csv_dir()
    session.url = url_for('soporte_vectorial')
    return render_template('soporte_vectorial.html', csv_list=csv_list)

# Get the uploaded files for soporte vectorial
@app.route("/soporte_vectorial", methods=['POST'])
def uploadFilesSoporteVectorial():
    return uploadFiles()

# --------- Funcionalidad ------------------

# Entrenar variables
def train(df, x, y, arbol=True):
    X = np.array(df[x])
    #X = x
    xNames = df[['Pregnancies', 
                       'Glucose', 
                       'BloodPressure', 
                       'SkinThickness', 
                       'Insulin', 
                       'BMI',
                       'DiabetesPedigreeFunction',
                       'Age']]
    #X = np.array(xNames)
    Y = np.array(df[[y]])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
    
    if(arbol):
        Pronostico = DecisionTreeRegressor(random_state=0)
    else:
        Pronostico = RandomForestRegressor(random_state=0)
    Pronostico.fit(X_train, Y_train)
    Y_Pronostico = Pronostico.predict(X_test)
    Valores = pd.DataFrame(Y_test, Y_Pronostico)
    Score = r2_score(Y_test, Y_Pronostico)

    return {'Pronostico':Pronostico,'Y_Pronostico':Y_Pronostico,'Valores':Valores,'Score':Score, 'X':x}

def obtener_pronostico(values, Pronostico):
    df = pd.DataFrame(values)
    return Pronostico.predict(df)[0]

@app.route("/ajax_parametros",methods=["POST","GET"])
def ajax_add():
    if request.method == 'POST':
        hidden_valorX = request.form['hidden_valorX']
        print(hidden_valorX)     
        msg = 'New record created successfully'  
    return redirect(url_for('arboles'))

def update_csv_dir():
    # Local  files
    for path in os.listdir('static/csv/'):
        # verificar archivo actual
        if path not in csv_list and os.path.isfile(os.path.join('static/csv/', path)):
            csv_list.append(path)

def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    name = uploaded_file.filename
    print("fileName: "+str(request.args.get('fileName')))
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
        # set the file path
        uploaded_file.save(file_path)
        # save the file
        session.fileSelected = name
    return redirect(session.url)

# -------- Graficas ---------------------------------------------------

# Callback for data
@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return gm(request.args.get('data'))

# Callback histogramas
@app.route('/hist', methods=['POST','GET'])
def histCall():
    return histogram(request.args.get('data'))

def histogram(name='melb_data.csv'):
    df = pd.read_csv('static/csv/'+name)

    fig = go.Figure()

    # Carga multiples graficos para poder cambiar
    for column in df.columns:
        fig.add_trace(
            go.Histogram(x=df[column])
        )

    # Crea de forma dinamica el boton para cada grafico
    def create_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    # Guarda cada boton para acceder, por medio de función lambda
    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            buttons=(list(df.columns.map(lambda column: create_button(column)))),
            )
        ]
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Callback HeatMap
@app.route('/hMap', methods=['POST','GET'])
def hMap():
    return hMapGraph(request.args.get('data'))

def hMapGraph(name='melb_data.csv'):
    df = pd.read_csv('static/csv/'+name)

    df = df.corr()

    fig = px.imshow(df, text_auto=True)
    """fig = go.Figure(data=go.Heatmap{
                        z=df,
                        x=df.columns,
                        y=df.rows,
                    }, hoverongaps = False)"""

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Line Plot Varianza de Cargas
@app.route('/varianza', methods=['POST','GET'])
def varCall():
    return varGraph(request.args.get('data'))

def varGraph(name='Hipoteca.csv'):
    df = pd.read_csv('static/csv/'+name)

    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(df)
    
    pca = PCA(n_components=10)
    pca.fit(MEstandarizada)

    df_temp = np.cumsum(pca.explained_variance_ratio_)

    fig = px.line(df_temp)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Scatter Plot 
@app.route('/scatt', methods=['POST','GET'])
def scattCall():
    return scattGraph(request.args.get('data'))

def scattGraph(name='Hipoteca.csv'):
    df = pd.read_csv('static/csv/'+name)

    fig = px.scatter_matrix(df, color=df.columns[-1])

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Small Scatter Plot 
@app.route('/scattSmall', methods=['POST','GET'])
def scattCall2():
    return scattGraph2(request.args.get('data'), request.args.get('valorX'), request.args.get('valorY'), request.args.get('color'))

def scattGraph2(name=session.fileSelected, x = 'gastos_comunes', y='vivienda', color='ingresos'):
    df = pd.read_csv('static/csv/'+name)

    #print(str(x)+str(y)+str(color))

    #fig = px.scatter_matrix(df, dimensions=[df.columns[1], df.columns[0]], color=df.columns[-1])
    #fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[-1])
    fig = px.scatter(df, x=x, y=y, color=color)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/testScatter', methods=['POST','GET'])
def scatterTest():
    filename = request.args.get('fileName')
    x = request.args.get('valorX')
    y = request.args.get('valorY')
    color = request.args.get('color')

    print(str(x)+str(y)+str(color))

# ------ Para tablas -------------------------------------------------

# Normal
@app.route('/_get_table')
def get_table():
    fileName = request.args.get('fileName')
    df = pd.read_csv('static/csv/'+fileName)
    session.fileSelected = fileName

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Descripcion de Datos
@app.route('/describe')
def describe():
    fileName = request.args.get('fileName')
    df = pd.read_csv('static/csv/'+fileName)
    df = df.describe()

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Matriz de Correlacion
@app.route('/corr')
def corr():
    fileName = request.args.get('fileName')
    df = pd.read_csv('static/csv/'+fileName)
    df = df.corr()

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Tabla para carga de componentes PCA ACP
@app.route('/carComp')
def carComp():
    fileName = request.args.get('fileName')
    df = pd.read_csv('static/csv/'+fileName)
    
    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(df)
    
    pca = PCA(n_components=10)
    pca.fit(MEstandarizada)
    df = pd.DataFrame(abs(pca.components_), columns=df.columns)

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Tabla para mostrar variables Arboles

def gm(country='United Kingdom'):
    df = pd.DataFrame(px.data.gapminder())

    fig = px.line(df[df['country']==country], x="year", y="gdpPercap", title=country)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON