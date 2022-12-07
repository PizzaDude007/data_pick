
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import json

import plotly
import plotly.express as px
import plotly.graph_objects as go

import os
from os.path import join, dirname, realpath

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/csv'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# -------- PAGINAS ---------------------------------------------------

# Root URL
@app.route('/')
def index():
    return render_template('index.html')#,  graphJSON=gm())

@app.route('/eda')
def eda():
    return render_template('eda.html')

# Get the uploaded files
@app.route("/eda", methods=['POST'])
def uploadFilesEDA():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
          # save the file
      return redirect(url_for('eda'))

@app.route('/pca')
def pca():
    return render_template('pca.html')

# Get the uploaded files
@app.route("/pca", methods=['POST'])
def uploadFilesPCA():
      # get the uploaded file
      uploaded_file = request.files['file']
      if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
           uploaded_file.save(file_path)
          # save the file
      return redirect(url_for('pca'))


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

    fig = px.histogram(df, x=df.columns[2])

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

@app.route('/varianza', methods=['POST','GET'])
def varCall():
    return varGraph(request.args.get('data'))

def varGraph(name='melb_data.csv'):
    df = pd.read_csv('static/csv/'+name)

    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(df)
    
    pca = PCA(n_componentes=10)
    pca.fit(MEstandarizada)

    fig = px.line(np.cumsum(pca.explained_variance_ratio_), 
            x='Número de componentes',
            y='Varianza Acumulada')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


# ------ Para tablas -------------------------------------------------

# Normal
@app.route('/_get_table')
def get_table():
    fileName = request.args.get('fileName')
    df = pd.read_csv('static/csv/'+fileName)

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

def gm(country='United Kingdom'):
    df = pd.DataFrame(px.data.gapminder())

    fig = px.line(df[df['country']==country], x="year", y="gdpPercap", title=country)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON