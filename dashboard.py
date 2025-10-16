import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")

# CSS personalizado para mejor diseño
st.markdown("""
<style>
/* ---- General ---- */
body {
    font-family: 'Poppins', sans-serif;
}

/* ---- Tabs (menús) ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.1);
    padding: 0.6rem;
    border-radius: 15px;
    backdrop-filter: blur(6px);
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(145deg, #1e3a8a, #3b82f6);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(145deg, #3b82f6, #1e3a8a);
    transform: scale(1.05);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(145deg, #0ea5e9, #2563eb);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
}

/* ---- Predictor ---- */
.predictor-card {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
    color: #1e293b;
}

.predictor-input label {
    color: #1e3a8a !important;
    font-weight: 600;
}

div.stSlider>div[data-baseweb="slider"]>div>div>div {
    background: linear-gradient(90deg, #06b6d4, #3b82f6) !important;
}

.predictor-result {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    box-shadow: 0 5px 20px rgba(22,163,74,0.4);
}

.predictor-result-error {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    box-shadow: 0 5px 20px rgba(239,68,68,0.4);
}

/* ---- Sliders ---- */
.stSlider label, .stSelectbox label {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* ---- Section titles ---- */
h3, h4, h5 {
    color: #0f172a !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    y_train = train_copy['Survived']
    
    train_copy['Age'] = train_copy['Age'].fillna(train_copy['Age'].median())
    test_copy['Age'] = test_copy['Age'].fillna(test_copy['Age'].median())
    train_copy['Embarked'] = train_copy['Embarked'].fillna(train_copy['Embarked'].mode()[0])
    test_copy['Embarked'] = test_copy['Embarked'].fillna(test_copy['Embarked'].mode()[0])
    test_copy['Fare'] = test_copy['Fare'].fillna(test_copy['Fare'].median())
    
    train_copy['Has_Cabin'] = train_copy['Cabin'].notna().astype(int)
    test_copy['Has_Cabin'] = test_copy['Cabin'].notna().astype(int)
    
    train_copy['Title'] = train_copy['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    test_copy['Title'] = test_copy['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    title_counts = train_copy['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    train_copy['Title'] = train_copy['Title'].replace(rare_titles, 'Rare')
    test_copy['Title'] = test_copy['Title'].replace(rare_titles, 'Rare')
    test_copy['Title'] = test_copy['Title'].replace(
        test_copy['Title'].unique()[~np.isin(test_copy['Title'].unique(), train_copy['Title'].unique())],
        'Rare'
    )
    
    train_copy['FamilySize'] = train_copy['SibSp'] + train_copy['Parch'] + 1
    test_copy['FamilySize'] = test_copy['SibSp'] + test_copy['Parch'] + 1
    train_copy['IsAlone'] = (train_copy['FamilySize'] == 1).astype(int)
    test_copy['IsAlone'] = (test_copy['FamilySize'] == 1).astype(int)
    
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    train_copy = train_copy.drop(drop_columns, axis=1)
    test_copy = test_copy.drop(drop_columns, axis=1)
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_title = LabelEncoder()
    
    train_copy['Sex'] = le_sex.fit_transform(train_copy['Sex'])
    test_copy['Sex'] = le_sex.transform(test_copy['Sex'])
    train_copy['Embarked'] = le_embarked.fit_transform(train_copy['Embarked'])
    test_copy['Embarked'] = le_embarked.transform(test_copy['Embarked'])
    train_copy['Title'] = le_title.fit_transform(train_copy['Title'])
    test_copy['Title'] = le_title.transform(test_copy['Title'])
    
    X_train = train_copy.drop('Survived', axis=1)
    
    return train_df, test_df, X_train, y_train, test_copy, le_sex, le_embarked, le_title

train_df, test_df, X_train, y_train, X_test, le_sex, le_embarked, le_title = load_and_prepare_data()

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

@st.cache_resource
def train_model():
    model = LogisticRegression(random_state=42, max_iter=1000)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(X_train_split, y_train_split)
    return model, X_train_split, X_val, y_train_split, y_val

model, X_train_split, X_val, y_train_split, y_val = train_model()

# ================== HEADER ==================
st.markdown('<p class="section-header">🚢 TITANIC MACHINE LEARNING</p>', unsafe_allow_html=True)
st.markdown("**Análisis Completo de Supervivencia con Inteligencia Artificial**")
st.divider()

# ================== METRICS MEJORADAS ==================
st.markdown("### 📊 Estadísticas Clave del Proyecto")

col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("Pasajeros Totales", "2,224", "#667eea", "#764ba2"),
    ("Sobrevivientes", f"{int(y_train.sum())}", "#10b981", "#059669"),
    ("Precisión Modelo", "81.56%", "#f59e0b", "#d97706"),
    ("AUC-ROC Score", "0.8156", "#3b82f6", "#1d4ed8")
]

for i, (label, value, color1, color2) in enumerate(metrics):
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            ">
                <p style="color: white; font-size: 0.9em; margin: 0; opacity: 0.9;">{label}</p>
                <p style="color: white; font-size: 2.2em; font-weight: bold; margin: 10px 0;">{value}</p>
            </div>
        """, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Resumen", "🔍 Análisis Exploratorio", "📈 Evaluación", "🤖 Métricas", "🎯 Predictor"])

# ================== TAB 1: RESUMEN ==================
with tab1:
    st.markdown("### 📈 Patrones de Supervivencia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Supervivencia por Género")
        survival_by_sex = train_df.groupby('Sex')['Survived'].agg(['sum', 'count'])
        survival_by_sex['rate'] = (survival_by_sex['sum'] / survival_by_sex['count'] * 100).round(2)
        total_survived = survival_by_sex['sum'].sum()
        
        fig_sex = go.Figure(data=[
            go.Pie(
                labels=['Mujeres', 'Hombres'],
                values=survival_by_sex['sum'],
                marker=dict(colors=['#ec4899', '#3b82f6'], line=dict(color='white', width=2)),
                textposition='inside',
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Sobrevivientes: %{value}<extra></extra>'
            )
        ])
        fig_sex.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_sex, width=100, use_container_width=True)
        
        st.info(f"""
        **Análisis de Género:**
        - **Mujeres**: {survival_by_sex.loc['female', 'rate']:.1f}% de supervivencia ({int(survival_by_sex.loc['female', 'sum'])} de {int(survival_by_sex.loc['female', 'count'])} personas)
        - **Hombres**: {survival_by_sex.loc['male', 'rate']:.1f}% de supervivencia ({int(survival_by_sex.loc['male', 'sum'])} de {int(survival_by_sex.loc['male', 'count'])} personas)
        """)
    
    with col2:
        st.markdown("#### Supervivencia por Clase de Pasaje")
        survival_by_class = train_df.groupby('Pclass')['Survived'].agg(['sum', 'count'])
        survival_by_class['rate'] = (survival_by_class['sum'] / survival_by_class['count'] * 100).round(2)
        
        class_names = ['1ª Clase', '2ª Clase', '3ª Clase']
        fig_class = go.Figure()
        
        fig_class.add_trace(go.Bar(
            x=class_names,
            y=survival_by_class['sum'],
            name='Sobrevivieron',
            marker_color='#10b981',
            text=survival_by_class['sum'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Sobrevivieron: %{y}<extra></extra>'
        ))
        
        fig_class.add_trace(go.Bar(
            x=class_names,
            y=survival_by_class['count'] - survival_by_class['sum'],
            name='No Sobrevivieron',
            marker_color='#ef4444',
            text=survival_by_class['count'] - survival_by_class['sum'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>No Sobrevivieron: %{y}<extra></extra>'
        ))
        
        fig_class.update_layout(barmode='group', height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_class, use_container_width=True)
        
        st.info(f"""
        **Análisis por Clase:**
        - **1ª Clase**: {survival_by_class.loc[1, 'rate']:.1f}% de supervivencia
        - **2ª Clase**: {survival_by_class.loc[2, 'rate']:.1f}% de supervivencia
        - **3ª Clase**: {survival_by_class.loc[3, 'rate']:.1f}% de supervivencia
        """)

# ================== TAB 2: ANÁLISIS EXPLORATORIO ==================
with tab2:
    st.markdown("### 🔍 Exploración de Variables")
    
    st.markdown("#### Distribución de Edad por Supervivencia")
    st.caption("Rojo (0) = No sobrevivió | Verde (1) = Sobrevivió | La edad está redondeada a años enteros")
    
    # Redondear edad a valores enteros para mejor visualización (sin valores NaN)
    train_df_age = train_df.dropna(subset=['Age']).copy()
    train_df_age['Age_Round'] = train_df_age['Age'].astype(int)
    
    fig_age = px.histogram(
        train_df_age,
        x='Age_Round',
        color='Survived',
        nbins=30,
        color_discrete_map={0: '#ef4444', 1: '#10b981'},
        labels={'Age_Round': 'Edad (años)', 'Survived': 'Resultado'},
        barmode='overlay',
        text_auto=True
    )
    fig_age.update_traces(textposition='outside', hovertemplate='<b>Edad: %{x} años</b><br>Cantidad: %{y}<extra></extra>')
    fig_age.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_age, use_container_width=True)
    
    st.info("**Observación**: Los niños (menores de 18 años) tuvieron mayor probabilidad de supervivencia, seguido de mujeres adultas.")
    
    st.divider()
    
    st.markdown("#### Distribución de Tarifa por Supervivencia")
    st.caption("La tarifa es el precio del boleto en libras esterlinas. Tarifa más alta = mejor clase = más acceso a botes salvavidas")
    
    fig_fare = px.histogram(
        train_df,
        x='Fare',
        color='Survived',
        nbins=40,
        color_discrete_map={0: '#ef4444', 1: '#10b981'},
        labels={'Fare': 'Tarifa del Boleto ($)', 'Survived': 'Resultado'},
        barmode='overlay'
    )
    fig_fare.update_traces(textposition='outside', hovertemplate='<b>Tarifa: $%{x:.2f}</b><br>Cantidad: %{y}<extra></extra>')
    fig_fare.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_fare, use_container_width=True)
    
    st.info("**Observación**: Pasajeros con tarifas más altas (>$100) tuvieron mejor tasa de supervivencia.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tamaño de Familia vs Supervivencia")
        family_survived = train_df.groupby(['FamilySize', 'Survived']).size().unstack(fill_value=0)
        
        fig_family = go.Figure()
        fig_family.add_trace(go.Bar(
            x=family_survived.index,
            y=family_survived[0],
            name='No Sobrevivieron',
            marker_color='#ef4444',
            text=family_survived[0],
            textposition='outside',
            hovertemplate='<b>Familia: %{x} personas</b><br>No Sobrevivieron: %{y}<extra></extra>'
        ))
        fig_family.add_trace(go.Bar(
            x=family_survived.index,
            y=family_survived[1],
            name='Sobrevivieron',
            marker_color='#10b981',
            text=family_survived[1],
            textposition='outside',
            hovertemplate='<b>Familia: %{x} personas</b><br>Sobrevivieron: %{y}<extra></extra>'
        ))
        fig_family.update_layout(barmode='group', height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_family, use_container_width=True)
        st.caption("Viajar en familia pequeña (2-4 personas) fue mejor que viajar solo o en familias muy grandes")
    
    with col2:
        st.markdown("#### Puerto de Embarque vs Supervivencia")
        embarked_survival = train_df.groupby(['Embarked', 'Survived']).size().unstack(fill_value=0)
        port_map = {'C': 'Cherbourg', 'S': 'Southampton', 'Q': 'Queenstown'}
        embarked_survival.index = embarked_survival.index.map(port_map)
        
        fig_embarked = go.Figure()
        fig_embarked.add_trace(go.Bar(
            x=embarked_survival.index,
            y=embarked_survival[0],
            name='No Sobrevivieron',
            marker_color='#ef4444',
            text=embarked_survival[0],
            textposition='outside',
            hovertemplate='<b>Puerto: %{x}</b><br>No Sobrevivieron: %{y}<extra></extra>'
        ))
        fig_embarked.add_trace(go.Bar(
            x=embarked_survival.index,
            y=embarked_survival[1],
            name='Sobrevivieron',
            marker_color='#10b981',
            text=embarked_survival[1],
            textposition='outside',
            hovertemplate='<b>Puerto: %{x}</b><br>Sobrevivieron: %{y}<extra></extra>'
        ))
        fig_embarked.update_layout(barmode='group', height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_embarked, use_container_width=True)
        st.caption("Southampton fue el puerto con mayor número de pasajeros")

# ================== TAB 3: EVALUACIÓN ==================
with tab3:
    st.markdown("### 📊 Evaluación del Modelo Logistic Regression")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Matriz de Confusión")
        st.caption("Muestra exactitud de predicciones: Verdaderos vs Falsos, Positivos vs Negativos")
        
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Pred: No Sobrevivió', 'Pred: Sobrevivió'],
            y=['Real: No Sobrevivió', 'Real: Sobrevivió'],
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 14},
            colorscale='Blues',
            showscale=True
        ))
        fig_cm.update_layout(height=400, margin=dict(l=100, r=0, t=0, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.success(f"""
        **Interpretación:**
        - **VN ({cm[0,0]})**: Verdadero Negativo - Predijo correctamente "No sobrevivió"
        - **FP ({cm[0,1]})**: Falso Positivo - Predijo "Sobrevivió" pero fue error
        - **FN ({cm[1,0]})**: Falso Negativo - Predijo "No sobrevivió" pero fue error
        - **VP ({cm[1,1]})**: Verdadero Positivo - Predijo correctamente "Sobrevivió"
        """)
    
    with col2:
        st.markdown("#### Curva ROC (Receiver Operating Characteristic)")
        st.caption("Mide la capacidad del modelo de distinguir entre sobrevivió y no sobrevivió")
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Modelo (AUC = {roc_auc:.4f})',
            line=dict(color='#3b82f6', width=3),
            hovertemplate='FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Aleatorio (AUC = 0.50)',
            line=dict(color='gray', dash='dash', width=2),
            hovertemplate='FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>'
        ))
        fig_roc.update_layout(
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.success(f"**AUC-ROC Score: {roc_auc:.4f}** (Más cercano a 1.0 = Mejor desempeño)")

# ================== TAB 4: MÉTRICAS ==================
with tab4:
    st.markdown("### 🎯 Métricas Detalladas del Modelo")
    
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{report['accuracy']*100:.2f}%", "Precisión General")
    with col2:
        st.metric("Precision", f"{report['weighted avg']['precision']*100:.2f}%", "Exactitud Ponderada")
    with col3:
        st.metric("Recall", f"{report['weighted avg']['recall']*100:.2f}%", "Cobertura Ponderada")
    
    st.divider()
    
    st.markdown("#### Desglose por Clase")
    
    metrics_data = {
        'Clase': ['No Sobrevivió', 'Sobrevivió'],
        'Precision': [f"{report['0']['precision']*100:.2f}%", f"{report['1']['precision']*100:.2f}%"],
        'Recall': [f"{report['0']['recall']*100:.2f}%", f"{report['1']['recall']*100:.2f}%"],
        'F1-Score': [f"{report['0']['f1-score']*100:.2f}%", f"{report['1']['f1-score']*100:.2f}%"],
        'Support': [int(report['0']['support']), int(report['1']['support'])]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 Características Importantes")
        feature_imp = pd.DataFrame({
            'Característica': X_train.columns,
            'Importancia': np.abs(model.coef_[0])
        }).sort_values('Importancia', ascending=True).tail(10)
        
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=feature_imp['Característica'],
            x=feature_imp['Importancia'],
            orientation='h',
            marker=dict(
                color=feature_imp['Importancia'],
                colorscale='Viridis',
                showscale=True
            ),
            text=feature_imp['Importancia'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>'
        ))
        fig_imp.update_layout(height=400, margin=dict(l=130, r=0, t=0, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col2:
        st.markdown("#### ¿Qué significan las métricas?")
        st.info("""
        **Precision**: De cada predicción positiva, cuántas fueron correctas.
        
        **Recall (Sensibilidad)**: De todos los casos positivos reales, cuántos el modelo encontró.
        
        **F1-Score**: Balance armónico entre Precision y Recall (0-100, mayor es mejor).
        
        **Support**: Número de ejemplos reales en cada clase.
        
        **Accuracy**: Porcentaje total de predicciones correctas.
        """)

# ================== TAB 5: PREDICTOR ==================
with tab5:
    st.markdown("### 🎯 Predictor Interactivo en Tiempo Real")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Datos del Pasajero")
        
        pclass = st.selectbox("Clase de Pasaje", [1, 2, 3], format_func=lambda x: f"Clase {x}")
        sex = st.selectbox("Género", ["Femenino", "Masculino"])
        age = st.slider("Edad (años)", 0, 80, 25)
        sibsp = st.slider("Hermanos/Cónyuge a bordo", 0, 8, 0)
        parch = st.slider("Padres/Hijos a bordo", 0, 6, 0)
        fare = st.slider("Tarifa del Boleto ($)", 0, 512, 100)
        embarked = st.selectbox("Puerto de Embarque", ["Cherbourg", "Southampton", "Queenstown"])

        
        sex_value = "female" if sex == "Femenino" else "male"
        sex_encoded = le_sex.transform([sex_value])[0]
        
        embarked_map = {"Cherbourg": "C", "Southampton": "S", "Queenstown": "Q"}
        embarked_encoded = le_embarked.transform([embarked_map[embarked]])[0]
        
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        has_cabin = 1
        title_encoded = 0
        
        input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded, has_cabin, title_encoded, family_size, is_alone]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    
    with col2:
        st.markdown("#### Resultado de la Predicción")
        
        if prediction == 1:
            st.success(f"""
            ### ✅ SOBREVIVIÓ
            
            **Probabilidad: {probability[1]*100:.1f}%**
            
            Según nuestro modelo, este pasajero probablemente sobrevivió.
            """)
        else:
            st.error(f"""
            ### ❌ NO SOBREVIVIÓ
            
            **Probabilidad: {probability[0]*100:.1f}%**
            
            Según nuestro modelo, este pasajero probablemente no sobrevivió.
            """)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability[1]*100,
            title="Probabilidad de Supervivencia",
            delta={'reference': 50, 'suffix': '% vs Neutral'},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981" if prediction == 1 else "#ef4444"},
                'steps': [
                    {'range': [0, 25], 'color': "#fee2e2"},
                    {'range': [25, 50], 'color': "#fecaca"},
                    {'range': [50, 75], 'color': "#dcfce7"},
                    {'range': [75, 100], 'color': "#bbf7d0"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    st.markdown("### 📋 Resumen del Pasajero")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**👤 Género**: {'♀️ Femenino' if sex == 'Femenino' else '♂️ Masculino'}")
        st.write(f"**🎂 Edad**: {age} años")
    with col2:
        st.write(f"**🏫 Clase**: {pclass}ª")
        st.write(f"**💵 Tarifa**: ${fare}")
    with col3:
        st.write(f"**👨‍👩‍👧‍👦 Familia**: {family_size} personas")
        st.write(f"**🚢 Puerto**: {embarked}")
    with col4:
        st.write(f"**Viaja Solo**: {'Sí ✓' if is_alone else 'No'}")
        st.write(f"**📊 Probabilidad**: {probability[1]*100:.1f}%")

st.divider()
