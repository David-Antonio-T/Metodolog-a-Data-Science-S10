from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle
import os
import tempfile
import matplotlib.pyplot as plt

app = Flask(__name__)

# Cargar el modelo, el scaler y las columnas desde el archivo pkl
with open('titanic_model_scaler_columns.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    scaler = data['scaler']
    expected_columns = data['columns']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No se subió ningún archivo"

    file = request.files['file']
    if file.filename == '':
        return "No se seleccionó ningún archivo"

    if file and file.filename.endswith('.csv'):
        # Usar una carpeta temporal
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, file.filename)
        file.save(filepath)

        # Leer el archivo CSV subido
        data = pd.read_csv(filepath)

        # Preprocesar el archivo subido
        data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

        # Separar las columnas numéricas y categóricas
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Imputar valores faltantes en columnas numéricas (con la mediana)
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        # Imputar valores faltantes en columnas categóricas (con la moda)
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

        # Convertir columnas categóricas en variables dummy
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Asegurarse de que las columnas coincidan con las esperadas
        missing_cols = set(expected_columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[expected_columns]

        # Escalar los datos con el scaler cargado
        data_scaled = scaler.transform(data)

        # Hacer predicciones con el modelo cargado
        predictions = model.predict(data_scaled)

        # Añadir las predicciones al dataset original
        data['Sobrevivientes'] = predictions

        # Calcular contadores y porcentajes
        total_passengers = len(data)
        survived_count = sum(predictions)
        not_survived_count = total_passengers - survived_count
        survived_percentage = (survived_count / total_passengers) * 100
        not_survived_percentage = (not_survived_count / total_passengers) * 100

        # Estadísticas por grupo
        survived_data = data[data['Sobrevivientes'] == 1]
        not_survived_data = data[data['Sobrevivientes'] == 0]

        statistics = {
            'Sobrevivientes': {
                'Promedio de Edad': survived_data['Age'].mean(),
                'Promedio de Familiares a Bordo (SibSp + Parch)': (
                            survived_data['SibSp'] + survived_data['Parch']).mean(),
                'Promedio de Tarifa': survived_data['Fare'].mean()
            },
            'Fallecidos': {
                'Promedio de Edad': not_survived_data['Age'].mean(),
                'Promedio de Familiares a Bordo (SibSp + Parch)': (
                            not_survived_data['SibSp'] + not_survived_data['Parch']).mean(),
                'Promedio de Tarifa': not_survived_data['Fare'].mean()
            }
        }

        result = {
            'Total': total_passengers,
            'Sobrevivientes': survived_count,
            'Fallecidos': not_survived_count,
            'Sobrevivientes %': survived_percentage,
            'Fallecidos %': not_survived_percentage,
            'Statistics': statistics
        }

        # Gráfico de distribución por sexo
        male_survived = survived_data['Sex_male'].sum()
        female_survived = len(survived_data) - male_survived

        male_not_survived = not_survived_data['Sex_male'].sum()
        female_not_survived = len(not_survived_data) - male_not_survived

        labels = ['Sobrevivientes Masculinos', 'Sobrevivientes Femeninos', 'Fallecidos', 'Fallecidas']
        values = [male_survived, female_survived, male_not_survived, female_not_survived]

        # Gráfico de distribución por sexo
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(labels, values,
                      color=['#4CAF50', '#81C784', '#F44336', '#E57373'])

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom', fontsize=10, ha='center')

        ax.set_xticklabels(labels, fontsize=10, rotation=0)

        ax.set_xlabel('Categoría')
        ax.set_ylabel('Número de Pasajeros')


        graph_sex_path = os.path.join(temp_dir, 'sex_distribution_plot.png')
        plt.tight_layout()
        plt.savefig(graph_sex_path)
        plt.close()

        # Gráfico original: Sobrevivientes vs Fallecidos
        fig, ax = plt.subplots()
        bars = ax.bar(['Sobrevivientes', 'Fallecidos'], [len(survived_data), len(not_survived_data)], color=['green', 'red'])

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', fontsize=10, ha='center')

        ax.set_xlabel('Categoría')
        ax.set_ylabel('Número de Pasajeros')
        ax.set_title('Sobrevivientes vs Fallecidos')

        graph_survival_path = os.path.join(temp_dir, 'survival_plot.png')
        plt.tight_layout()
        plt.savefig(graph_survival_path)
        plt.close()

        return render_template('result.html', result=result, graph_url='/plot_survival', graph_sex_url='/plot_sex')


@app.route('/plot_survival')
def plot_survival():
    temp_dir = tempfile.gettempdir()
    graph_path = os.path.join(temp_dir, 'survival_plot.png')
    return send_file(graph_path, mimetype='image/png')


@app.route('/plot_sex')
def plot_sex():
    temp_dir = tempfile.gettempdir()
    graph_sex_path = os.path.join(temp_dir, 'sex_distribution_plot.png')
    return send_file(graph_sex_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
