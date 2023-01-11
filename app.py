
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import pandas as pd
import numpy as np
import json

import plotly
import plotly.express as px
import plotly.graph_objects as go
import igraph
from igraph import Graph, EdgeSeq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import io
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import pandas as pd
import numpy as np
import json

import plotly
import plotly.express as px
import plotly.graph_objects as go
import igraph
from igraph import Graph, EdgeSeq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import io
import os
from os.path import join, dirname, realpath

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, pairwise_distances_argmin_min, classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

from kneed import KneeLocator
#from sklearn.externals import joblib

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/csv'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

app.config['TEMPLATES_AUTO_RELOAD'] = True

repo = 'static/csv/'
#repo = 'https://raw.githubusercontent.com/PizzaDude007/data_pick/main/static/csv/'
csv_list = ['DiabeticRetinopathy.csv', 'DiabeticRetinopathy.csv', 'WDBCOriginal.csv', 'diabetes.csv', 'melb_data.csv']

# session.file

class session:
    fileSelected = 'diabetes.csv'
    url = ''
    fileEDA = 'melb_data.csv'
    filePCA = 'Hipoteca.csv'
    fileArboles = 'diabetes.csv'
    fileSegmentacion = 'diabetes.csv'
    fileSoporteVectorial = 'diabetes.csv'
    arbolesRes = {}
    bosquesRes = {}
    svmRes = {}
    isRegression = True
    MEstandarizada = None
    MParticional = None
    pd = pd.read_csv(repo+fileSelected)
    firstPass = True

class svm_multiple:
    linear = {}
    poly = {}
    rbf = {}
    sigmoid = {}
    empty = True

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
@app.route("/eda", methods=['POST', 'GET'])
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

    df = pd.read_csv(repo+session.fileSelected)

    return render_template('pca.html', table=df, pd=pd, nameData=session.fileSelected, csv_list=csv_list)

# Get the uploaded files for pca
@app.route("/pca", methods=['POST', 'GET'])
def uploadFilesPCA():
    return uploadFiles()

@app.route('/arboles')
def arboles():
    update_csv_dir()
    session.url = url_for('arboles')
    fileName = request.args.get('fileName')
    x = request.args.getlist('valorX') # Variables Predictoras
    y = request.args.get('valorY') # Variable a Pronosticas
    max_d = request.args.get('max_depth') # Profundidad Maxima
    min_s = request.args.get('min_samples_split') # Minimo de muestras para dividir
    min_l = request.args.get('min_samples_leaf') # Minimo de muestras por hoja

    df = pd.read_csv(repo+session.fileSelected)
    pronostico = ''
    dparams = []
    dictOpciones = {}

    print('Internal: '+str(session.fileSelected))
    print('External: '+str(fileName))
    
    if (x is not None and y is not None):
        print(str(x))
        #print(y)
        try:
            session.arbolesRes = train(df, x, y, max_d, min_s, min_l, True, session.isRegression)
        except:
            print('Error en craga de arboles')
        if (max_d is not None and min_s is not None and min_l is not None):
            dparams = [max_d, min_s, min_l]
    elif ('X' in session.arbolesRes.keys() and session.firstPass):
        #if (x is None or (x is not None and x[0] in session.arbolesRes['X'] and y == session.arbolesRes['Y'])):
        for name in session.arbolesRes['X']:
            print(name+": "+str(request.args.get(name)))
            if(request.args.get(name) is not None):
                dictOpciones[name] = [float(request.args.get(name))]
                print(dictOpciones[name])
        pronostico = str(obtener_pronostico(dictOpciones, session.arbolesRes['Pronostico']))
        #dfPronostico = pd.DataFrame.from_dict(dictOpciones)
        #pronostico = str(session.arbolesRes['Pronostico'].predict(dfPronostico))

    
    return render_template('arboles.html', table=df, nameData=session.fileSelected, res=session.arbolesRes, pronostico=pronostico, csv_list=csv_list, params=dparams, isRegression=session.isRegression)

# Get the uploaded files
@app.route("/arboles", methods=['POST', 'GET'])
def uploadFilesArboles():
    return uploadFiles()

@app.route('/bosques')
def bosques():
    update_csv_dir()
    session.url = url_for('bosques')
    fileName = request.args.get('fileName')
    x = request.args.getlist('valorX') # Variables Predictoras
    y = request.args.get('valorY') # Variable a Pronosticas
    max_d = request.args.get('max_depth') # Profundidad Maxima
    min_s = request.args.get('min_samples_split') # Minimo de muestras para dividir
    min_l = request.args.get('min_samples_leaf') # Minimo de muestras por hoja

    df = pd.read_csv(repo+session.fileSelected)
    pronostico = ''
    dparams = []
    dictOpciones = {}

    print('Internal: '+str(session.fileSelected))
    print('External: '+str(fileName))

    if (x is not None and y is not None):
        print(str(x))
        #print(y)
        try:
            session.bosquesRes = train(df, x, y, max_d, min_s, min_l, False, session.isRegression)
        except:
            print('Error en carga de bosques')
        if (max_d is not None and min_s is not None and min_l is not None):
            dparams = [max_d, min_s, min_l]
    elif ('X' in session.bosquesRes.keys()):
        #if (x is None or (x is not None and x[0] in session.bosquesRes['X'] and y == session.bosquesRes['Y'])):
        for name in session.bosquesRes['X']:
            print(name+": "+str(request.args.get(name)))
            if(request.args.get(name) is not None):
                dictOpciones[name] = [float(request.args.get(name))]
                print(dictOpciones[name])
        pronostico = str(obtener_pronostico(dictOpciones, session.bosquesRes['Pronostico']))
    
    
    return render_template('bosques.html', table=df, nameData=session.fileSelected, res=session.bosquesRes, pronostico=pronostico, csv_list=csv_list, params=dparams, isRegression=session.isRegression)

# Get the uploaded files
@app.route("/bosques", methods=['POST', 'GET'])
def uploadFilesBosques():
    return uploadFiles()

# Segmentacion
@app.route('/segmentacion')
def segmentacion():
    update_csv_dir()
    session.url = url_for('segmentacion')
    columnDrop = request.args.get('drop')
    df = pd.read_csv(repo+session.fileSelected)

    if columnDrop is not None and columnDrop != '':
        df.drop(['columnDrop'], axis=1, inplace=True)
    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(df)
    session.MEstandarizada = MEstandarizada
    

    MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    session.MParticional = MParticional

    return render_template('segmentacion.html', nameData=session.fileSelected, csv_list=csv_list)

# Get the uploaded files for segmentacion
@app.route("/segmentacion", methods=['POST', 'GET'])
def uploadFilesSegmentacion():
    return uploadFiles()

# Soporte vectorial
@app.route('/soporte_vectorial')
def soporte_vectorial():
    update_csv_dir()
    session.url = url_for('soporte_vectorial')
    fileName = request.args.get('fileName')
    x = request.args.getlist('valorX_PR') # Variables Predictoras
    y = request.args.get('valorY_PR') # Variable a Pronosticas
    kernel = request.args.get('kernel') # Kernel
    max_d = request.args.get('max_depth') # Profundidad Maxima
    min_s = request.args.get('min_samples_split') # Minimo de muestras para dividir
    min_l = request.args.get('min_samples_leaf') # Minimo de muestras por hoja

    df = pd.read_csv(repo+session.fileSelected)

    pronostico = ''
    dparams = []
    dictOpciones = {}

    print('Internal: '+str(session.fileSelected))
    print('External: '+str(fileName))

    if (x is not None and y is not None):
        print(str(x))
        #print(y)
        try:
            session.svmRes = train(df, x, y, max_d, min_s, min_l, svm=True, kernel=kernel)
        except:
            print('Error en carga de SVM')
        if (max_d is not None and min_s is not None and min_l is not None):
            dparams = [max_d, min_s, min_l]
    elif ('X' in session.svmRes.keys()): #and x[0] in session.svmRes['X'] and kernel == session.svmRes['kernel'] and y == session.svmRes['Y']):
        #if(x is None or (x is not None and x[0] in session.svmRes['X'] and kernel == session.svmRes['kernel'] and y == session.svmRes['Y'])):
        for name in session.svmRes['X']:
            print(name+": "+str(request.args.get(name)))
            if(request.args.get(name) is not None):
                dictOpciones[name] = [float(request.args.get(name))]
                print(dictOpciones[name])
        pronostico = str(obtener_pronostico(dictOpciones, session.svmRes['Pronostico']))


    return render_template('soporte_vectorial.html', table=df, nameData=session.fileSelected, res=session.svmRes, pronostico=pronostico, csv_list=csv_list, params=dparams, isRegression=session.isRegression)

# Get the uploaded files for soporte vectorial
@app.route("/soporte_vectorial", methods=['POST', 'GET'])
def uploadFilesSoporteVectorial():
    return uploadFiles()

# --------- Funcionalidad ------------------

# Entrenar variables
def train(df, x, y, max_d=0, min_s=0, min_l=0,arbol=True, regresion=True, svm=False, kernel='linear'):
    X = np.array(df[x])
    #X = x
    #X = np.array(xNames)
    Y = np.array(df[[y]])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
    
    # Revisa si es arbol bosque o maquina de soporte vectorial, ademas de clasificacion o regresion
    if(arbol and regresion and not svm): # Arbol - Regresion
        if(max_d is not None and min_s is not None and min_l is not None):
            Pronostico = DecisionTreeRegressor(max_depth=int(max_d), min_samples_split=int(min_s), min_samples_leaf=int(min_l), random_state=0)
        else:
            Pronostico = DecisionTreeRegressor(random_state=0)
    elif(arbol and not regresion and not svm): # Arbol - Clasificacion
        if(max_d is not None and min_s is not None and min_l is not None):
            Pronostico = DecisionTreeClassifier(max_depth=int(max_d), min_samples_split=int(min_s), min_samples_leaf=int(min_l), random_state=0)
        else:
            Pronostico = DecisionTreeClassifier(random_state=0)
    elif(not arbol and regresion and not svm): # Bosque - Regresion
        if(max_d is not None and min_s is not None and min_l is not None):
            Pronostico = RandomForestRegressor(max_depth=int(max_d), min_samples_split=int(min_s), min_samples_leaf=int(min_l), random_state=0)
        else:
            Pronostico = RandomForestRegressor(random_state=0)
    elif(not arbol and not regresion and not svm): # Bosque - Clasificacion
        if(max_d is not None and min_s is not None and min_l is not None):
            Pronostico = RandomForestClassifier(max_depth=int(max_d), min_samples_split=int(min_s), min_samples_leaf=int(min_l), random_state=0)
        else:
            Pronostico = RandomForestClassifier(random_state=0)
    elif(svm): # Maquina de soporte vectorial
        #Pronostico = SVC(kernel=kernel, random_state=0, probability=True)
        Pronostico = SVC(kernel=kernel, random_state=0)
        
    Pronostico.fit(X_train, Y_train.ravel())

    #joblib.dump(Pronostico, 'model.pkl')
    Y_Pronostico = Pronostico.predict(X_test)
    Valores = pd.DataFrame(Y_test, Y_Pronostico)
    roc = None
    if svm:
        Score = accuracy_score(Y_test, Y_Pronostico)
        #roc = roc_curve(Y_test, Pronostico.predict_proba(X_test)[:,1], pos_label=1)
        roc = roc_curve(Y_test, Y_Pronostico, pos_label=1)
    else:
        Score = r2_score(Y_test, Y_Pronostico)
    

    return {'Pronostico':Pronostico,'Y_Pronostico':Y_Pronostico,'Valores':Valores,'Score':Score, 'X':x, 'Y':y, 'roc':roc, 'X_test':X_test, 'Y_test':Y_test, 'kernel':kernel}

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
    for path in os.listdir(repo):
        # verificar archivo actual
        if path not in csv_list and os.path.isfile(os.path.join(repo, path)):
            csv_list.append(path)

def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    name = uploaded_file.filename
    isPronostico = request.args.get('isPronostico')
    if(isPronostico is not None):
        session.isRegression = isPronostico
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
    df = pd.read_csv(repo+name)

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
    df = pd.read_csv(repo+name)

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

def varGraph(name=session.fileSelected):
    df = pd.read_csv(repo+name)

    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(df)
    
    pca = PCA(n_components=10)
    pca.fit(MEstandarizada)

    df_temp = np.cumsum(pca.explained_variance_ratio_)

    fig = px.line(df_temp)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Line Plot Elbow Method
@app.route('/elbow', methods=['POST','GET'])
def elbowCall():
    return elbowGraph(request.args.get('data'))

def elbowGraph(name=session.fileSelected):
    session.fileSelected = name
    #df = pd.read_csv(repo+name)

    MEstandarizada = session.MEstandarizada
    df_temp = pd.DataFrame(MEstandarizada)

    SSE = []
    for i in range(2,10):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(df_temp)
        SSE.append(km.inertia_)

    k1 = KneeLocator(range(2,10), SSE, curve="convex", direction="decreasing")

    fig = px.line(SSE, x=range(2,10), y=SSE, title='Elbow Method', labels={'x':'Cantidad de Clusters K', 'y':'SSE'})
    fig.add_vline(x=k1.elbow, line_width=3, line_dash="dash", line_color="red")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# ROC 
@app.route('/roc', methods=['POST','GET'])
def rocCall():
    return rocGraph(request.args.get('data'))

def rocGraph(name=session.fileSelected):
    session.fileSelected = name
    #df = pd.read_csv(repo+name)
    
    fpr, tpr, thresholds = session.svmRes['roc']
    roc_auc = auc(fpr, tpr)
    print('ROC: '+str(roc_auc))

    fig = px.area(x=fpr, y=tpr, title=f'Curva ROC (AUC={roc_auc:.4f})',labels={'x':'FPR', 'y':'TPR'})
    fig.add_shape(type="line", line_dash="dash", x0=0, x1=1, y0=0, y1=1)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/cluster', methods=['POST','GET'])
def clusterCall():
    return clusterGraph(request.args.get('data'))

def clusterGraph(name=session.fileSelected):
    session.fileSelected = name
    df = pd.read_csv(repo+name)
    
    MEstandarizada = session.MEstandarizada
    MParticional = session.MParticional
    #MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
    #MParticional.predict(MEstandarizada)
    
    fig = px.scatter_3d(df, x=MEstandarizada[:,0], y=MEstandarizada[:,1], z=MEstandarizada[:,2], color=MParticional.labels_)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Scatter Plot 
@app.route('/scatt', methods=['POST','GET'])
def scattCall():
    return scattGraph(request.args.get('data'))

def scattGraph(name=session.fileSelected):
    df = pd.read_csv(repo+name)

    fig = px.scatter_matrix(df, color=df.columns[-1])

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# Small Scatter Plot 
@app.route('/scattSmall', methods=['POST','GET'])
def scattCall2():
    return scattGraph2(request.args.get('data'), request.args.get('valorX'), request.args.get('valorY'), request.args.get('color'))

def scattGraph2(name=session.fileSelected, x = session.pd.columns[0], y=session.pd.columns[1], color=session.pd.columns[-1]):
    df = pd.read_csv(repo+name)

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

"""
@app.route('/tree', methods=['POST','GET'])
def crearArbol(name=session.fileSelected):
    # 2YYuezJtkaidcZoh5wbk
    # ipython -c "import plotly; plotly.tools.set_credentials_file(username='PieterVDW', api_key='2YYuezJtkaidcZoh5wbk')"
    session.arbolesRes['']
    df = pd.read_csv(repo+name)
    fig = px.treemap(session.arbolesRes['Pronostico'], )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
"""

@app.route('/tree.png')
def tree_png():
    #plt.figure()
    #plot_tree(session.arbolesRes['Pronostico'], feature_names=session.arbolesRes['X'])
    #plt.savefig('tree.eps', format='eps',bbox_inches="tight")

    fig = plt.figure(figsize=(20,20)) #Figure()
    plot_tree(session.arbolesRes['Pronostico'], feature_names=session.arbolesRes['X'])
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/forest.png')
def forest_png():
    fig = plt.figure(figsize=(20,20)) #Figure()
    Estimador = session.bosquesRes['Pronostico'].estimators_[99]
    plot_tree(Estimador, feature_names=session.bosquesRes['X'])
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

# No se ocupa
"""
@app.route('/svm.png')
def svm_png():
    fig, ax = plt.subplots()

    if(svm_multiple.empty):
        df = pd.read_csv(repo+session.fileSelected)
        x = session.svmRes['X']
        y = session.svmRes['Y']
        svm_multiple.linear = train(df, x, y, kernel='linear', svm=True)
        svm_multiple.poly = train(df, x, y, kernel='poly', svm=True)
        svm_multiple.rbf = train(df, x, y, kernel='rbf', svm=True)
        svm_multiple.sigmoid = train(df, x, y, kernel='sigmoid', svm=True)
        
    RocCurveDisplay.from_estimator(svm_multiple.linear['Pronostico'],
                            svm_multiple.linear['X_test'],
                            svm_multiple.linear['Y_test'],
                            ax = ax,
                            name='Lineal')                                   
    RocCurveDisplay.from_estimator(svm_multiple.poly['Pronostico'],
                                        svm_multiple.poly['X_test'],
                                        svm_multiple.poly['Y_test'],
                                        ax = ax,
                                        name='Polinomial')
    RocCurveDisplay.from_estimator(svm_multiple.rbf['Pronostico'],
                                        svm_multiple.rbf['X_test'],
                                        svm_multiple.rbf['Y_test'],
                                        ax = ax,
                                        name='RBF')
    RocCurveDisplay.from_estimator(svm_multiple.sigmoid['Pronostico'],
                                        svm_multiple.sigmoid['X_test'],
                                        svm_multiple.sigmoid['Y_test'],
                                        ax = ax,
                                        name='Sigmoide')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
"""

# ------ Para tablas -------------------------------------------------

# Normal
@app.route('/_get_table')
def get_table():
    fileName = request.args.get('fileName')
    df = pd.read_csv(repo+fileName)
    session.fileSelected = fileName

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Descripcion de Datos
@app.route('/describe')
def describe():
    fileName = request.args.get('fileName')
    df = pd.read_csv(repo+fileName)
    df = df.describe()

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Matriz de Correlacion
@app.route('/corr')
def corr():
    fileName = request.args.get('fileName')
    df = pd.read_csv(repo+fileName)
    df = df.corr()

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Tabla para carga de componentes PCA ACP
@app.route('/carComp')
def carComp():
    fileName = request.args.get('fileName')
    df = pd.read_csv(repo+fileName)
    
    Estandarizar = StandardScaler()
    MEstandarizada = Estandarizar.fit_transform(df)
    
    pca = PCA(n_components=10)
    pca.fit(MEstandarizada)
    df = pd.DataFrame(abs(pca.components_), columns=df.columns)

    return jsonify(#number_elements=a * b,
                   my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Tabla para Obtencion de Centroides
@app.route('/carCentroides')
def carCentroides():
    fileName = request.args.get('fileName')
    columnDrop = request.args.get('drop')

    df = pd.read_csv(repo+session.fileSelected)

    MEstandarizada = session.MEstandarizada 

    #MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
    #MParticional.predict(MEstandarizada)

    MParticional = session.MParticional
    
    df['clusterP'] = MParticional.labels_
    df = df.groupby('clusterP').mean()

    return jsonify(my_table=json.loads(df.to_json(orient="split"))["data"],
                   columns=[{"title": str(col)} for col in json.loads(df.to_json(orient="split"))["columns"]])

# Tabla para mostrar variables Arboles

def gm(country='United Kingdom'):
    df = pd.DataFrame(px.data.gapminder())

    fig = px.line(df[df['country']==country], x="year", y="gdpPercap", title=country)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
