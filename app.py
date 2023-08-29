import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# Página Home
def home():
    col1, col2 , col3 = st.columns(3)
    with col2:
        st.image('grafico_logo.png')
        st.markdown('---')
        st.title('Stock Market App')
        st.markdown('---')

#Pagina Panorama de Mercado
def panorama():
    st.title('Stock Market Review')
    st.markdown(date.today().strftime('%d/%m/%Y'))

    st.subheader('Around the world')

    # Dicionário de Indices x Ticker do YFinance
    dict_tickers = {
                'Bovespa':'^BVSP', 
                'S&P500':'^GSPC',
                'NASDAQ':'^IXIC', 
                'DAX':'^GDAXI', 
                'FTSE 100':'^FTSE',
                'Cruid Oil': 'CL=F',
                'Gold':'GC=F',
                'BITCOIN':'BTC-USD',
                'ETHEREUM':'ETH-USD'
                }

    # Montagem do Dataframe de informaçções dos indices
    df_info = pd.DataFrame({'Market asset': dict_tickers.keys(),'Ticker': dict_tickers.values()})
    
    df_info['Ult. Valor'] = ''
    df_info['%'] = ''
    count =0
    with st.spinner('Downloading prices...'):
        for ticker in dict_tickers.values():
            cotacoes = yf.download(ticker, period='5d')['Adj Close']
            variacao = ((cotacoes.iloc[-1]/cotacoes.iloc[-2])-1)*100
            df_info['Ult. Valor'][count] = round(cotacoes.iloc[-1],2)
            df_info['%'][count] =round(variacao,2)
            count += 1

    # Apresentação do Dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(df_info['Market asset'][0], value=df_info['Ult. Valor'][0], delta=str(df_info['%'][0]) + '%')
        st.metric(df_info['Market asset'][1], value=df_info['Ult. Valor'][1], delta=str(df_info['%'][1]) + '%')
        st.metric(df_info['Market asset'][2], value=df_info['Ult. Valor'][2], delta=str(df_info['%'][2]) + '%')

    with col2:
        st.metric(df_info['Market asset'][3], value=df_info['Ult. Valor'][3], delta=str(df_info['%'][3]) + '%')
        st.metric(df_info['Market asset'][4], value=df_info['Ult. Valor'][4], delta=str(df_info['%'][4]) + '%')
        st.metric(df_info['Market asset'][5], value=df_info['Ult. Valor'][5], delta=str(df_info['%'][5]) + '%')

    with col3:
        st.metric(df_info['Market asset'][6], value=df_info['Ult. Valor'][6], delta=str(df_info['%'][6]) + '%')
        st.metric(df_info['Market asset'][7], value=df_info['Ult. Valor'][7], delta=str(df_info['%'][7]) + '%')
        st.metric(df_info['Market asset'][8], value=df_info['Ult. Valor'][8], delta=str(df_info['%'][8]) + '%')

    # Comportamento ao longo do dia
    st.markdown('---')

    st.subheader('Behavior during the day')

    # Seleção de Indices e coleta de dados
    lista_indices = ['IBOV', 'S&P500', 'NASDAQ']

    indice = st.selectbox('Select Index', lista_indices)

    if indice == 'IBOV':
        indice_diario = yf.download('^BVSP', period='1d', interval='5m')
    if indice == 'S&P500':
        indice_diario = yf.download('^GSPC', period='1d', interval='5m')
    if indice == 'NASDAQ':
        indice_diario = yf.download('^IXIC', period='1d', interval='5m')

    # Grafico de CandleStick
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Candlestick(x=indice_diario.index,
                        open=indice_diario['Open'],
                        high=indice_diario['High'],
                        low=indice_diario['Low'],
                        close=indice_diario['Close'])])
    fig.update_layout(title=indice, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

    # Seleção de Ações
    lista_acoes = ['PETR4.SA', 'VALE3.SA', 'EQTL3.SA', 'CSNA3.SA']

    acao = st.selectbox('Select stock', lista_acoes)

    hist_acao = yf.download(acao, period='1d', interval='5m')

    # Grafico de CandleStick
    fig = go.Figure(data=[go.Candlestick(x=hist_acao.index,
                        open=hist_acao['Open'],
                        high=hist_acao['High'],
                        low=hist_acao['Low'],
                        close=hist_acao['Close'])])
    fig.update_layout(title=acao, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)


# Página Retornos Mensais
def mapa_mensal():
    st.title('Monthly Returns Analysis')

    # Seleção das opções
    with st.expander('Chose', expanded=True):
        opcao = st.radio('Select', ['Index', 'Stocks'])
    
    if opcao == 'Index':
        with st.form(key='form_indice'):
            indices_dict = {'Bovespa': '^BVSP', 'S&P 500': '^GSPC', 'Bitcoin USD': 'BTC-USD'}
            ticker = st.selectbox('Index', list(indices_dict.keys()))
            analisar = st.form_submit_button('Analysis')
    else:
        with st.form(key='form_acoes'):
            acoes_dict = {'PETR4': 'PETR4.SA', 'EQTL3': 'EQTL3.SA', 'VALE3': 'VALE3.SA'}
            ticker = st.selectbox('Stocks', list(acoes_dict.keys()))
            analisar = st.form_submit_button('Analysis')

    if analisar:
        data_inicial = '1999-01-01'
        data_final = str(date.today())

        if opcao == 'Index':
            ticker_symbol = indices_dict[ticker]
        else:
            ticker_symbol = acoes_dict[ticker]

        # Fetch data using yfinance
        retornos = yf.download(ticker_symbol, start=data_inicial, end=data_final, progress=False)

        # Separar e agrupar os anos e meses
        retornos['Month'] = retornos.index.month
        retornos['Year'] = retornos.index.year
        retorno_mensal = retornos.groupby(['Year', 'Month'])['Close'].mean().unstack()

        # Heatmap
        retorno_mensal_pct = retorno_mensal.pct_change()

        # Heatmap for percentage variations
        cmap = sns.color_palette('RdYlGn', 50)
        fig, ax = plt.subplots(figsize=(15, 11))
        sns.heatmap(retorno_mensal_pct, cmap='RdYlGn', annot=True, fmt='.2%', center=0, cbar=False,
                    linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title(ticker + ' - Monthly Percentage Variations', fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='12')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize='12')
        ax.xaxis.tick_top()  # x axis on top
        plt.ylabel('')
        st.pyplot(fig)


        stats = pd.DataFrame(retorno_mensal.mean(), columns=['Average'])
        stats['Mean'] = retorno_mensal.median()
        stats['Biggest'] = retorno_mensal.max()
        stats['Smallest'] = retorno_mensal.min()

        positive_years = (retorno_mensal_pct > 0).sum()
        negative_years = (retorno_mensal_pct < 0).sum()
        total_years = retorno_mensal_pct.count()

        stats['Positives'] = (positive_years / total_years) 
        stats['Negatives'] = (negative_years / total_years) 

        # Plot statistics
        stats_a = stats[['Average', 'Mean', 'Biggest', 'Smallest']]
        stats_a = stats_a.transpose()

        fig, ax = plt.subplots(figsize=(15, 3))
        cmap_stats = sns.color_palette('RdYlBu', as_cmap=True)  # Use 'RdYlBu' colormap
        sns.heatmap(stats_a, cmap=cmap_stats, annot=True, fmt='.1f', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                    linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='11')
        st.pyplot(fig)

        stats_b = stats[['Positives', 'Negatives']]
        stats_b = stats_b.transpose()

        fig, ax = plt.subplots(figsize=(15, 2))
        sns.heatmap(stats_b, annot=True, fmt='.1%', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                    linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='11')
        st.pyplot(fig)

# Página Fundamentos
def fundamentos():
    import fundamentus as fd # Biblioteca do site Fundamentus

    st.title('Fundamentalist information')

    lista_tickers = fd.list_papel_all()

    comparar = st.checkbox('Compare two stocks')

    col1, col2 = st.columns(2)

    with col1:
        with st.expander('Stock 1', expanded=True):
            papel1 = st.selectbox('Select the stock', lista_tickers)
            info_papel1 = fd.get_detalhes_papel(papel1)
            st.write('**Company:**', info_papel1['Empresa'][0])
            st.write('**Sector:**', info_papel1['Setor'][0])
            st.write('**Subsector:**', info_papel1['Subsetor'][0])
            st.write('**Market value:**', f"R$ {float(info_papel1['Valor_de_mercado'][0]):,.2f}")
            st.write('**Net Worth:**', f"R$ {float(info_papel1['Patrim_Liq'][0]):,.2f}")
            st.write('**ROE:**', f"{info_papel1['ROE'][0]}")
            st.write("**Gross debt / Shareholders' equity:**", f"{info_papel1['Div_Br_Patrim'][0]}")
            st.write('**P/E:**', f"{float(info_papel1['PL'][0]):,.2f}")
            st.write('**Dividend Yield:**', f"{info_papel1['Div_Yield'][0]}")

    if comparar:
        with col2:
            with st.expander('Ativo 2', expanded=True):
                papel2 = st.selectbox('Selecione o 2º Papel', lista_tickers)
                info_papel2 = fd.get_detalhes_papel(papel2)
                st.write('**Company:**', info_papel2['Empresa'][0])
                st.write('**Sector:**', info_papel2['Setor'][0])
                st.write('**Subsector:**', info_papel2['Subsetor'][0])
                st.write('**Market value:**', f"R$ {float(info_papel2['Valor_de_mercado'][0]):,.2f}")
                st.write('**Net Worth:**', f"R$ {float(info_papel2['Patrim_Liq'][0]):,.2f}")
                st.write('**ROE:**', f"{info_papel2['ROE'][0]}")
                st.write("**Gross debt / Shareholders' equity:**", f"{info_papel2['Div_Br_Patrim'][0]}")
                st.write('**P/E:**', f"{float(info_papel2['PL'][0]):,.2f}")
                st.write('**Dividend Yield:**', f"{info_papel2['Div_Yield'][0]}")

# Função principal
def main():
    st.sidebar.image('grafico_logo.png', width=200)
    st.sidebar.title('Stock Market App')
    st.sidebar.markdown('---')
    lista_menu=['Home', 'Market Overview', 'Monthly Returns','Fundamentals']
    escolha = st.sidebar.radio('Escolha a opção', lista_menu)

    if escolha =='Home':
        home()
    if escolha == 'Market Overview':
        panorama()
    if escolha =='Monthly Returns':
        mapa_mensal()
    if escolha == 'Fundamentals':
        fundamentos()


if __name__ == "__main__":
    main()
