# Trabalho de Conclusão de Curso apresentado para obtenção do título de especialista em Data Science & Analytics – 2025 
# Nome: Stephanie Escorcio Franke

## Data Wrangling ##

#%% Importando os pacotes

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

#%% Importando os bancos de dados

dados_orig = pd.read_excel("BD-LLMs-AA-all.xlsx")
dados_models = pd.read_excel("models-data_2025-06-06.xlsx")

#%% Visualizando informações básicas do dataset

# Configurar para printar objetos no console

pd.set_option("display.max.columns", None)

# Informações detalhadas sobre as variáveis

dados_orig.info()

# object = variável de texto
# int ou float = variável numérica (métrica)
# category = variável categórica (qualitativa)

#%% Alterando os nomes das variáveis

# Renomeando as variáveis pela sua posição (criando um objeto)

dados_llms = dados_orig.rename(columns={dados_orig.columns[0]: 'api_provider',
                                         dados_orig.columns[1]: 'model',
                                         dados_orig.columns[2]: 'context_window',
                                         dados_orig.columns[3]: 'function_calling',
                                         dados_orig.columns[4]: 'json_mode',
                                         dados_orig.columns[5]: 'license',
                                         dados_orig.columns[6]: 'openai_compatible',
                                         dados_orig.columns[7]: 'api_id',
                                         dados_orig.columns[8]: 'footnotes',
                                         dados_orig.columns[9]: 'intelligence_index',
                                         dados_orig.columns[10]: 'mmlu-pro',
                                         dados_orig.columns[11]: 'gpqa',
                                         dados_orig.columns[12]: 'humanity',
                                         dados_orig.columns[13]: 'livecodebench',
                                         dados_orig.columns[14]: 'scicode',
                                         dados_orig.columns[15]: 'humaneval',
                                         dados_orig.columns[16]: 'math',
                                         dados_orig.columns[17]: 'aime',
                                         dados_orig.columns[18]: 'multilingual',
                                         dados_orig.columns[19]: 'chatbot',
                                         dados_orig.columns[20]: 'blended_price',
                                         dados_orig.columns[21]: 'input_price',
                                         dados_orig.columns[22]: 'output_price',
                                         dados_orig.columns[23]: 'median_tokens',
                                         dados_orig.columns[24]: 'p5_tokens',
                                         dados_orig.columns[25]: 'p25_tokens',
                                         dados_orig.columns[26]: 'p75_tokens',
                                         dados_orig.columns[27]: 'p95_tokens',
                                         dados_orig.columns[28]: 'median_ttft',
                                         dados_orig.columns[29]: 'firstanswer_latency',
                                         dados_orig.columns[30]: 'p5_ttft',
                                         dados_orig.columns[31]: 'p25_ttft',
                                         dados_orig.columns[32]: 'p75_ttft',
                                         dados_orig.columns[33]: 'p95_ttft',
                                         dados_orig.columns[34]: 'response_time',
                                         dados_orig.columns[35]: 'reasoning_time',
                                         dados_orig.columns[36]: 'further_analysis',
                                         dados_orig.columns[37]: 'parallel_queries',
                                         dados_orig.columns[38]: 'prompt_length'})


#%% Selecionando variáveis de interesse

var_creators = dados_models[['Model Name', 'Release Date', 'Model Creator Name', 'Model Slug']]

var_creators.rename(columns={var_creators.columns[0]: 'model_name',
                           var_creators.columns[1]: 'release_date',
                           var_creators.columns[2]: 'creator',
                           var_creators.columns[3]: 'model_slug'},
                  inplace=True)

#%% Adicionar nova variável aplicando critérios detalhados por meio de condições

# Substituição direta de nomes de variáveis
var_creators['model_name'] = var_creators['model_name'].str.replace(' Instruct', '', regex=False)
var_creators['model_name'] = var_creators['model_name'].str.replace('A22B ', '', regex=False)

var_creators.loc[var_creators['model_name'] == "Claude 3.5 Sonnet (Oct '24)", 'model_name'] = 'Claude 3.5 Sonnet (Oct)'
var_creators.loc[var_creators['model_name'] == "Claude 3.5 Sonnet (June '24)", 'model_name'] = 'Claude 3.5 Sonnet (June)'
var_creators.loc[var_creators['model_name'] == 'Qwen2.5 32B', 'model_name'] = 'Qwen2.5 Instruct 32B'
var_creators.loc[var_creators['model_name'] == 'Jamba', 'model_name'] = 'Jamba Instruct'
var_creators.loc[var_creators['model_name'] == "Gemini 2.0 Pro Experimental (Feb '25)", 'model_name'] = 'Gemini 2.0 Pro ExperimentaL'
var_creators.loc[var_creators['model_name'] == 'Llama 3.1 Nemotron Ultra 253B v1 (Reasoning)', 'model_name'] = 'Llama 3.1 Nemotron Ultra 253B'
var_creators.loc[var_creators['model_name'] == "DeepSeek V3 0324 (Mar '25)", 'model_name'] = "DeepSeek V3 (Mar' 25)"

# Substituir todos os '.' (pontos) da variável 'api_id' por '-' (hífens)
dados_llms['api_id'] = dados_llms['api_id'].str.replace('.', '-', regex=False)

# Faz o merge com base na igualdade entre dados_llms['model'] e var_creators['model_name']
dados_llms = dados_llms.merge(
    var_creators[['model_name', 'model_slug']],
    how='left',
    left_on='model',
    right_on='model_name'
)

# Substitui 'api_id' apenas se houve correspondência ('model_slug' não nulo)
dados_llms['api_id'] = dados_llms['model_slug'].combine_first(dados_llms['api_id'])

# Remove a coluna auxiliar 'model_slug' e 'model_name' vinda do var_creators
dados_llms.drop(columns=['model_name', 'model_slug'], inplace=True)

# Função para buscar correspondência de texto e retornar o valor das variáveis correspondente

def buscar_info_modelo_completo(row, df_ref):
    model_text = str(row['model']).lower().strip()
    api_id_text = str(row['api_id']).lower().strip()

    for _, ref_row in df_ref.iterrows():
        model_name = str(ref_row['model_name']).lower().strip()
        model_slug = str(ref_row['model_slug']).lower().strip()

        # Verifica se model ↔ model_name ou api_id ↔ model_slug (em qualquer direção)
        if (model_name in model_text or model_text in model_name or
            model_slug in api_id_text or api_id_text in model_slug):
            return pd.Series({
                'release_date': ref_row['release_date'],
                'creator': ref_row['creator'],
            })

    # Caso não encontre, retorna 'Não encontrado'
    return pd.Series({
        'release_date': 'Não encontrado',
        'creator': 'Não encontrado',
    })

# Aplica a função e cria as 4 novas colunas no dados_llms
dados_llms[['release_date', 'creator']] = dados_llms.apply(
    lambda row: buscar_info_modelo_completo(row, var_creators), axis=1
)

# Verificar que não há observações com 'creator': 'Não encontrado'
dados_llms['creator'].value_counts()

#%% Verificar se existem observações duplicadas

# Verifica duplicatas baseadas nas variáveis 'api_provider', 'model', 'parallel_queries' e 'prompt_length'
num_duplicatas = dados_llms.duplicated(subset=['api_provider', 'model', 'parallel_queries', 'prompt_length'], keep=False).sum()

# Exibe número de observações duplicadas
print(f"Número de observações duplicadas: {num_duplicatas}")

#%% Excluir variáveis que não são de interesse

dados_llms.drop(columns=['openai_compatible', 'api_id', 'footnotes', 'p5_tokens', 'p25_tokens', 'p75_tokens', 'p95_tokens', 'p5_ttft', 'p25_ttft', 'p75_ttft', 'p95_ttft', 'further_analysis'], inplace=True)

dados_llms.info()

# Variáveis sem observações com informação suficiente
dados_llms.drop(columns=['multilingual', 'chatbot', 'reasoning_time'], inplace=True)

# Variáveis que derivam de outras que já estão sendo analisadas
dados_llms.drop(columns=['intelligence_index', 'blended_price'], inplace=True)

#%% Reorganizar ordem das variáveis

vars_dados = list(dados_llms.columns)

# Lista das colunas que você quer mover
cols_to_move = ['parallel_queries', 'prompt_length', 'release_date', 'creator']

# Remove as colunas que serão movidas
for col in cols_to_move:
    vars_dados.remove(col)

# Encontra a posição da coluna 'model'
idx = vars_dados.index('model') + 1

# Insere as colunas a partir da posição após 'model'
for i, col in enumerate(cols_to_move):
    vars_dados.insert(idx + i, col)

# Reorganiza as colunas
dados_llms = dados_llms[vars_dados]

#%% Tratamento variável 'release_date'

# Converter release_date para datetime
dados_llms['release_date'] = pd.to_datetime(dados_llms['release_date'], errors='coerce')

# Criar release_year
dados_llms['release_year'] = dados_llms['release_date'].dt.year

# Criar days_since_release
hoje = pd.Timestamp(datetime.today())
dados_llms['days_since_release'] = (hoje - dados_llms['release_date']).dt.days

# Excluir variável 'release_date'
dados_llms.drop(columns=['release_date'], inplace=True)

#%% Conversão de variáveis

# Converter variáveis 'api_provider', 'parallel_queries', 'prompt_length', 'creator', 'function_calling', 'json_mode', 'license' e 'release_year' para category
dados_llms['api_provider'] = dados_llms['api_provider'].astype('category')
dados_llms['parallel_queries'] = dados_llms['parallel_queries'].astype('category')
dados_llms['prompt_length'] = dados_llms['prompt_length'].astype('category')
dados_llms['creator'] = dados_llms['creator'].astype('category')
dados_llms['function_calling'] = dados_llms['function_calling'].astype('category')
dados_llms['json_mode'] = dados_llms['json_mode'].astype('category')
dados_llms['license'] = dados_llms['license'].astype('category')
dados_llms['release_year'] = dados_llms['release_year'].astype('category')

# Converter a variável 'context_window' para int

def convert_str_to_int(x):
    x = str(x).strip().lower()
    if x.endswith('k'):
        return int(float(x[:-1]) * 1_000)
    elif x.endswith('m'):
        return int(float(x[:-1]) * 1_000_000)
    else:
        # Caso não tenha sufixo, tenta converter direto
        try:
            return int(float(x))
        except ValueError:
            return None  

dados_llms['context_window'] = dados_llms['context_window'].apply(convert_str_to_int)

# Converter variáveis 'input_price' e 'output_price' para float

cols_preco = ['input_price', 'output_price']

# Remove o símbolo de $ e converte para float
for col in cols_preco:
    dados_llms[col] = dados_llms[col].replace('[\$,]', '', regex=True).astype(float)

# Converter variáveis 'median_tokens' e 'median_ttft' para float
dados_llms['median_tokens'] = pd.to_numeric(dados_llms['median_tokens'], errors='coerce')
dados_llms['median_ttft'] = pd.to_numeric(dados_llms['median_ttft'], errors='coerce')


#%% Estatísticas gerais do banco de dados

dados_llms.info()

#%% Remover as observações com valores faltantes 'nan'

dados_llms.dropna(inplace=True)

# Novas informações sobre o dataset

dados_llms.info()

#%% Importar os pacotes

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
import prince
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.metrics import silhouette_score
from pandas.plotting import scatter_matrix

#%% ## CLUSTERIZAÇÃO NAS VARIÁVEIS QUANTITATIVAS ##

## CENÁRIOS ##
## Artificial Analysis - API providers ##

# CENÁRIO 1
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 100 tokens (short)

# CENÁRIO 2
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 1k tokens (medium)
        
# CENÁRIO 3
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 10k tokens (long)
        
# CENÁRIO 4
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: 100k tokens
        
# CENÁRIO 5
    # PROMPT OPTIONS:
        # Parallel Queries: Single
        # Prompt Length: Coding (1k tokens - medium)
        
# CENÁRIO 6
    # PROMPT OPTIONS:
        # Parallel Queries: Multiple
        # Prompt Length: 1k tokens (medium)

# Gerar bancos de dados para cada um dos cenários

df_single_short = dados_llms[(dados_llms['parallel_queries'] == 'single') & (dados_llms['prompt_length'] == 'short')]
df_single_medium = dados_llms[(dados_llms['parallel_queries'] == 'single') & (dados_llms['prompt_length'] == 'medium')]
df_single_long = dados_llms[(dados_llms['parallel_queries'] == 'single') & (dados_llms['prompt_length'] == 'long')]
df_single_100k = dados_llms[(dados_llms['parallel_queries'] == 'single') & (dados_llms['prompt_length'] == '100k')]
df_single_coding = dados_llms[(dados_llms['parallel_queries'] == 'single') & (dados_llms['prompt_length'] == 'medium_coding')]
df_multiple_medium = dados_llms[(dados_llms['parallel_queries'] == 'multiple') & (dados_llms['prompt_length'] == 'medium')]

#%% CENÁRIO 2

# Separar somente as variáveis quantitativas do banco de dados
df_quanti1 = df_single_medium.select_dtypes(include=['number'])

import plotly.graph_objects as go

# Estatísticas descritivas das variáveis
desc = df_quanti1.describe().T.round(2)

fig = go.Figure(data=[go.Table(
    columnwidth=[40] + [15] * (len(desc.columns)),  # Primeira mais larga, outras estreitas
    header=dict(
        values=["<b>Variável</b>"] + [f"<b>{col}</b>" for col in desc.columns],  # Negrito
        fill_color='white',
        font=dict(color='black', size=12),
        align='left',
        line_color='black'
    ),
    cells=dict(
        values=[desc.index] + [desc[col].tolist() for col in desc.columns],
        fill_color='white',
        font=dict(color='black'),
        align='left',
        line_color='black'
    )
)])

fig.update_layout(title="Estatísticas Descritivas")
fig.show()

#%% Análise exploratória dos dados
    
# Boxplot vertical para múltiplas variáveis numéricas

# Número de variáveis
num_vars = df_quanti1.shape[1]
cols = 4  # número de colunas no grid de subplots
rows = (num_vars // cols) + int(num_vars % cols != 0)

# Criar a figura
fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))

# Flatten os eixos para facilitar iteração
axes = axes.flatten()

# Plotar cada boxplot em um subplot separado
for i, col in enumerate(df_quanti1.columns):
    sns.boxplot(y=df_quanti1[col], ax=axes[i], color='lightblue')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Valor')
    
# Remover eixos vazios, se houver
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Mapa de correlação
plt.figure(figsize=(12,10))
sns.heatmap(df_quanti1.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.show()

# Matriz de dispersão
plt.figure(figsize=(16, 16))
scatter_matrix(df_quanti1, alpha=0.7, figsize=(16, 16), diagonal='kde', color='steelblue')
plt.suptitle('Scatter Matrix (Matriz de Dispersão)', y=0.9)
plt.show()
    

# Histograma

num_vars = df_quanti1.columns 

n_cols = 4  # 4 colunas por linha
n_rows = 4  # 4 linhas para 16 variáveis

plt.figure(figsize=(20, 16))  # Tamanho proporcional à grade

for i, col in enumerate(num_vars):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(df_quanti1[col], bins=20, color='darkturquoise', edgecolor='black')
    plt.title('', fontsize=10)  
    plt.xlabel(col, fontsize=16)  
    plt.ylabel('Frequência', fontsize=14)
    plt.xticks(rotation=45)

# Ajusta layout com mais espaço vertical entre os subplots
plt.tight_layout(h_pad=1)  
plt.show()

#%% Realizar a padronização por meio do Z-Score

# As variáveis estão em unidades de medidas distintas
df_quanti1_pad = df_quanti1.apply(zscore, ddof=1)

#%% Algoritmo Isolation Forest para detecção de outliers multivariados

from sklearn.ensemble import IsolationForest

# Aplicando Isolation Forest no DataFrame já normalizado
iso = IsolationForest(contamination='auto', random_state=42)
labels = iso.fit_predict(df_quanti1_pad)

# Contando o número de outliers identificados (-1) e de inliers (1)
n_outliers = (labels == -1).sum()
n_inliers = (labels == 1).sum()
total_amostras = n_outliers + n_inliers

# Calculando a taxa de contaminação efetiva
taxa_contaminacao_efetiva = n_outliers / total_amostras

print(f"Total de observações: {total_amostras}")
print(f"Número de outliers identificados: {n_outliers}")
print(f"Taxa de contaminação efetiva: {taxa_contaminacao_efetiva:.4f}")

# Criando uma cópia do DataFrame com coluna indicando outliers
df_quanti1_out = df_quanti1_pad.copy()
df_quanti1_out['is_outlier'] = labels == -1  # True para outliers

# Lista de índices considerados outliers
outlier_indices = df_quanti1_out[df_quanti1_out['is_outlier']].index.tolist()

# Printando as observações consideradas outliers em formato de tabela
print("Observações identificadas como outliers:")
print(df_quanti1_out[df_quanti1_out['is_outlier']])

# (Opcional) Ver número de outliers vs não-outliers
print("\nContagem de outliers:")
print(df_quanti1_out['is_outlier'].value_counts())

# Identificar os índices dos outliers
outlier_indices = df_quanti1_out[df_quanti1_out['is_outlier']].index


from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

# 1. Selecionar apenas os outliers (True)
df_outliers = df_quanti1_out[df_quanti1_out['is_outlier']].copy()

# Remover a coluna 'is_outlier' para clustering
df_outliers_features = df_outliers.drop(columns='is_outlier')

# 2. Aplicar DBSCAN nos outliers
dbscan = DBSCAN(eps=0.5, min_samples=3)  # Parâmetros ajustáveis
labels = dbscan.fit_predict(df_outliers_features)

# Adicionar os rótulos de cluster ao DataFrame
df_outliers['cluster'] = labels



# Remover as observações outliers diretamente do df_quanti1_pad
df_quanti1_pad.drop(index=outlier_indices, inplace=True)

df_single_medium.drop(index=outlier_indices, inplace=True)


#%% Cluster Hierárquico Aglomerativo: complete linkage + distância euclidiana quadrática

# Passo 2 - remover output_price porque p-valor > 0.05
df_quanti1_pad = df_quanti1_pad.drop(columns=['output_price', 'cluster'])

#%%
# Gerando o dendrograma

plt.figure(figsize=(16,8))
dend_sing = sch.linkage(df_quanti1_pad, method = 'complete', metric = 'sqeuclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 60)
plt.xticks([], [])
plt.xlabel('Modelos', fontsize=16)
plt.ylabel('Distância Euclidiana Quadrática', fontsize=16)
plt.axhline(y = 70, color = 'red', linestyle = '--')
plt.show()

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,15) 
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_quanti1_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,15)) 
plt.ylabel('WCSS', fontsize=16)
#plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,15) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(df_quanti1_pad)
    silhueta.append(silhouette_score(df_quanti1_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 15), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
#plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster K-means

# Vamos considerar 4 clusters
kmeans_final = KMeans(n_clusters = 4, init = 'random', random_state=100).fit(df_quanti1_pad)

# Gerando a variável para identificarmos os clusters gerados
kmeans_clusters = kmeans_final.labels_
df_single_medium['cluster'] = kmeans_clusters
df_quanti1_pad['cluster'] = kmeans_clusters
df_single_medium['cluster'] = df_single_medium['cluster'].astype('category')
df_quanti1_pad['cluster'] = df_quanti1_pad['cluster'].astype('category')

#%% ANOVA

# Analisando se todas as variáveis são significativas para a clusterização 

# Lista de variáveis independentes (exceto a coluna 'cluster')
variaveis = df_quanti1_pad.columns[:-1]
variavel_cluster = df_quanti1_pad.columns[-1]

# Lista para armazenar os resultados
resultados_anova = []

# Loop para aplicar ANOVA em cada variável
for var in variaveis:
    resultado = pg.anova(dv=var, between=variavel_cluster, data=df_quanti1_pad, detailed=True)
    resultado['variavel'] = var
    resultados_anova.append(resultado)

# Concatenar os resultados em um único DataFrame
tabela_anova = pd.concat(resultados_anova, ignore_index=True)

# Selecionar colunas desejadas e ordenar pelo p-valor
tabela_formatada = tabela_anova[['variavel', 'F', 'p-unc', 'np2']].sort_values(by='p-unc')

# Exibir a tabela formatada
print(tabela_formatada.to_string(index=False))


fig, ax = plt.subplots(figsize=(10, 1 + 0.5 * len(tabela_formatada)))
ax.axis('off')
tabela_plot = ax.table(cellText=tabela_formatada.round(4).values,
                       colLabels=tabela_formatada.columns,
                       cellLoc='center',
                       loc='center')
tabela_plot.auto_set_font_size(False)
tabela_plot.set_fontsize(10)
tabela_plot.scale(1, 1.5)
plt.title("Resultados do Teste F (ANOVA)", fontsize=14)
plt.tight_layout()
plt.show()

# Conclusão: todas as variáveis auxiliam na formação de pelo menos um dos clusters, ao nível de significância de 5%

#%% Quais são as características dos clusters em termos das variáveis métricas

# Selecionar variáveis numéricas
variaveis_metricas = df_single_medium.select_dtypes(include='number').columns.tolist()

# Incluir a variável 'cluster'
if 'cluster' not in variaveis_metricas:
    variaveis_metricas.append('cluster')

# Agrupar por cluster e calcular média
df_cluster_medias = df_single_medium[variaveis_metricas].groupby('cluster').mean().T  # Transpõe para clusters como colunas

# Criar figura
plt.figure(figsize=(10, len(df_cluster_medias) * 0.5))

# Criar a tabela
tabela = plt.table(
    cellText=df_cluster_medias.round(2).values,
    rowLabels=df_cluster_medias.index,
    colLabels=[f"Cluster {c}" for c in df_cluster_medias.columns],
    loc='center',
    cellLoc='center',
    rowLoc='center'
)

tabela.scale(1, 1.5)
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)

# Remover eixos
plt.axis('off')
plt.title("Média das Variáveis por Cluster", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

#%% Pie chart

# Contar modelos por cluster
cluster_counts = df_quanti1_pad['cluster'].value_counts().sort_index()

# Cores inspiradas em dark turquoise
colors = ['#40E0D0', '#48D1CC', '#20B2AA', '#66CDAA']  # expandir se houver mais clusters

# Rótulos com número e porcentagem
labels = [f'Cluster {i}\n{count} modelos\n({count / cluster_counts.sum():.1%})'
          for i, count in cluster_counts.items()]

# Criar gráfico
plt.figure(figsize=(6, 6))  # tamanho da figura geral
plt.pie(cluster_counts,
        labels=labels,
        colors=colors,
        radius=0.7,  # reduz tamanho do gráfico
        startangle=140,
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 11})  # aumenta fonte dos rótulos

# Título
plt.title('Distribuição de Modelos por Cluster', fontsize=14)
plt.tight_layout()
plt.show()

#%% Realizar uma ACM (Análise de Correspondência Múltipla) nas variáveis qualitativas

# Incluir o output da Clusterização (clusters) na ACM

# Separando somente as variáveis categóricas do banco de dados
df_quali1 = df_single_medium.select_dtypes(include=['category'])

df_quali1 = df_quali1.drop(columns=['parallel_queries', 'prompt_length'])

#%% Estatísticas descritivas univariadas

# Tabelas de frequências por variável
print(df_quali1.api_provider.value_counts())
print(df_quali1.creator.value_counts())
print(df_quali1.function_calling.value_counts())
print(df_quali1.json_mode.value_counts())
print(df_quali1.license.value_counts())
print(df_quali1.release_year.value_counts())
print(df_quali1.cluster.value_counts())

#%% Gráficos de barras

# Remover categorias "fantasmas" das variáveis categóricas
for col in df_quali1.select_dtypes(include='category').columns:
    df_quali1[col] = df_quali1[col].cat.remove_unused_categories()

# Layout: 4 gráficos por linha
cols = 4
rows = (len(df_quali1.columns) // cols) + int(len(df_quali1.columns) % cols != 0)

fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

vars_quali = df_quali1.columns[2:]

for i, col in enumerate(vars_quali):
    order = df_quali1[col].value_counts().index  # Ordenação decrescente
    sns.countplot(x=col, data=df_quali1, ax=axes[i], color='darkturquoise', order=order, width=0.5)
    
    axes[i].set_ylabel('Frequência', fontsize=14)
    axes[i].set_xlabel(col, fontsize=14)  
    axes[i].tick_params(axis='x', labelsize=11, rotation=0)  
    axes[i].set_title('')  # Remove título

# Remove gráficos vazios (caso haja mais subplots do que variáveis)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#%% Agrupar categorias raras

def agrupar_categorias_raras(col, threshold=0.018):
    freq = col.value_counts(normalize=True)
    categorias_raras = freq[freq < threshold].index
    return col.apply(lambda x: 'Outros' if x in categorias_raras else x)

df_quali1['api_provider'] = agrupar_categorias_raras(df_quali1['api_provider'], threshold=0.018)
df_quali1['creator'] = agrupar_categorias_raras(df_quali1['creator'], threshold=0.018)

#%% Tabelas de Contingência

pd.crosstab(df_quali1['function_calling'], df_quali1['license'])
pd.crosstab(df_quali1['json_mode'], df_quali1['license'])
pd.crosstab(df_quali1['release_year'], df_quali1['license'])

# Clusters
pd.crosstab(df_quali1['api_provider'], df_quali1['cluster'])
pd.crosstab(df_quali1['creator'], df_quali1['cluster'])
pd.crosstab(df_quali1['function_calling'], df_quali1['cluster'])
pd.crosstab(df_quali1['json_mode'], df_quali1['cluster'])
pd.crosstab(df_quali1['license'], df_quali1['cluster'])
pd.crosstab(df_quali1['release_year'], df_quali1['cluster'])


# Lista de combinações de variáveis para tabelas de contingência
combinacoes = [
    ('api_provider', 'cluster'),
    ('creator', 'cluster'),
    ('function_calling', 'cluster'),
    ('json_mode', 'cluster'),
    ('license', 'cluster'),
    ('release_year', 'cluster'),
]

# Parâmetros da grade de subplots
cols = 2
rows = (len(combinacoes) // cols) + int(len(combinacoes) % cols != 0)

fig, axes = plt.subplots(rows, cols, figsize=(24, 5.5 * rows))
axes = axes.flatten()

# Paleta inspirada em dark turquoise
turquoise_cmap = sns.light_palette("#40E0D0", as_cmap=True)

# Loop para gerar os heatmaps com fontes aumentadas
for i, (var1, var2) in enumerate(combinacoes):
    ct = pd.crosstab(df_quali1[var1], df_quali1[var2], normalize='columns')
    sns.heatmap(
        ct,
        annot=True,
        annot_kws={"fontsize": 14},  # maior fonte nos números
        cmap=turquoise_cmap,
        fmt='.2f',
        linewidths=0.5,
        cbar=False,
        ax=axes[i]
    )
    axes[i].set_title(f'{var1} vs {var2}', fontsize=16)
    axes[i].tick_params(axis='x', labelsize=14, rotation=45)
    axes[i].tick_params(axis='y', labelsize=14, rotation=0)

# Remove eixos vazios se houver
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%% Testes qui-quadrado para pares de variáveis

# Para ACM, todas as variáveis precisam apresentar associação estatisticamente significante com pelo menos uma outra variável da análise

# Vamos colocar como referência 'creator'

tabela1 = chi2_contingency(pd.crosstab(df_quali1["creator"],
                                       df_quali1["api_provider"]))
print(f"p-valor da estatística creator x api_provider: {round(tabela1[1], 4)}")

tabela2 = chi2_contingency(pd.crosstab(df_quali1["creator"], 
                                       df_quali1["function_calling"]))
print(f"p-valor da estatística creator x function_calling: {round(tabela2[1], 4)}")

tabela3 = chi2_contingency(pd.crosstab(df_quali1["creator"], 
                                       df_quali1["json_mode"]))
print(f"p-valor da estatística creator x json_mode: {round(tabela3[1], 4)}")

tabela4 = chi2_contingency(pd.crosstab(df_quali1["creator"], 
                                       df_quali1["license"]))
print(f"p-valor da estatística creator x license: {round(tabela4[1], 4)}")

tabela5 = chi2_contingency(pd.crosstab(df_quali1["creator"], 
                                       df_quali1["release_year"]))
print(f"p-valor da estatística creator x release_year: {round(tabela5[1], 4)}")

tabela6 = chi2_contingency(pd.crosstab(df_quali1["creator"], 
                                       df_quali1["cluster"]))
print(f"p-valor da estatística creator x cluster: {round(tabela6[1], 4)}")

# Todas apresentam associação significativa com pelo menos uma variável

#%% Elaborando a análise de correspondência múltipla

# Criando coordenadas para 10 dimensões (a seguir, verifica-se a viabilidade)
mca = prince.MCA(n_components=10).fit(df_quali1)

#%% Analisando os resultados

# Análise dos autovalores
tabela_autovalores = mca.eigenvalues_summary
print(tabela_autovalores)

# Inércia total da análise
print(mca.total_inertia_)

# Plotar apenas dimensões com inércia parcial superior à inércia total média
quant_dim = mca.J_ - mca.K_
print(mca.total_inertia_/quant_dim)

# Inércia total média = 0.14285
# Todas as 10 dimensões apresentam inércia parcial superior à inércia total média

#%% Obtendo as coordenadas-padrão das categorias das variáveis

coord_padrao = mca.column_coordinates(df_quali1)/np.sqrt(mca.eigenvalues_)
print(coord_padrao)

#%% Plotando o mapa perceptual (coordenadas-padrão) - com apenas as 3 primeiras dimensões!

# Primeiro passo: gerar um DataFrame detalhado

chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])

nome_categ=[]
for col in df_quali1:
    nome_categ.append(df_quali1[col].sort_values(ascending=True).unique())
    categorias = pd.DataFrame(nome_categ).stack().reset_index()

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'obs_z': chart[2],
                             'variavel': var_chart,
                             'categoria_id': categorias[0]})

# Segundo passo: gerar o gráfico de pontos

fig = px.scatter_3d(chart_df_mca, 
                    x='obs_x', 
                    y='obs_y', 
                    z='obs_z',
                    color='variavel',
                    text=chart_df_mca.categoria_id)
fig.show()

#%% Mapa Perceptual 2D

# Parte 1: Preparação do DataFrame para o Mapa Perceptual

chart = coord_padrao.reset_index()
var_chart = chart['index'].str.split('_', n=1, expand=True)[0]
categ_id = chart['index'].str.split('_', n=1, expand=True)[1]

chart_df_mca_2d = pd.DataFrame({
    'categoria_completa': chart['index'],
    'obs_x': chart[1],
    'obs_y': chart[2],
    'variavel': var_chart,
    'rotulo': categ_id
})


# Parte 1.5: FILTRAGEM PARA EXCLUIR PONTOS INDESEJADOS 

# Edite esta lista com todos os pontos que estão poluindo seu gráfico.
pontos_a_excluir = [
    'api_provider__Together.ai',
    'api_provider__Deepinfra',

]

# **Passo 2: Crie um novo DataFrame filtrado.**
# A função .isin() verifica quais categorias ESTÃO na lista.
# O operador ~ (til) inverte a seleção, ou seja, "que NÃO ESTÃO na lista".
df_filtrado = chart_df_mca_2d[~chart_df_mca_2d['categoria_completa'].isin(pontos_a_excluir)]


# Parte 2: Geração do Mapa Perceptual 2D (usando o DataFrame filtrado)

fig = px.scatter(df_filtrado,  
                 x='obs_x',
                 y='obs_y',
                 color='variavel',
                 text='rotulo',
                 title="Mapa Perceptual 2D (Pontos Relevantes)")


#Parte 3: Melhorias no Layout 

fig.update_traces(
    textfont_size=12,
    textposition='top center',
    marker=dict(opacity=0.8, size=10)
)
fig.update_layout(
    template='plotly_white',
    title_x=0.5,
    height=900,
    width=1200,
    xaxis_title="Ciclo de Lançamento e Adoção",
    yaxis_title="Plataforma de Adoção e Capacidade de Ferramentas",
    legend_title="<b>Variáveis</b>",
    font=dict(family="Arial, sans-serif", size=14, color="black"),
    xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
    yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black')
)

fig.show()