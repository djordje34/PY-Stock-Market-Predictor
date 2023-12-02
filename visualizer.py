from matplotlib import pyplot as plt
import pandas as pd
import mplfinance as mpf
import seaborn as sns
from preprocess import load_data
import datetime as dt
from tensorflow.keras.models import load_model
import visualkeras
from PIL import ImageFont

def plot_ltsm_model(path):
    model = load_model(path)
    font = ImageFont.truetype("arial.ttf", 12)
    visualkeras.layered_view(model, legend=True, to_file='gen/model_architecture.png', font=font)

def plot_komp_cnt(name_counts):
    name_counts.plot(kind='bar', figsize=(10, 6), color='maroon')
    plt.title('Број тачака (дана) у којима је сачувана цена акције')
    plt.xlabel('Име компаније')
    plt.ylabel('Број тачака')
    plt.savefig("gen/brdana_komp.png")
    plt.close()
    
def plot_closing_price(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Closing_Price'], label='Цена затварања акција', color='maroon')
    plt.title('Peugeot - цена затварања акција у функцији времена')
    plt.xlabel('Датум')
    plt.ylabel('Цена затварања акција')
    plt.legend()
    plt.savefig("gen/cena_zatv_akc.png")
    plt.close()

def plot_daily_price_changes(df):
    plt.figure(figsize=(10, 6))
    temp = df
    temp['Daily_Price_Change'] = temp['Closing_Price'].diff()
    temp['Daily_Price_Change'].hist(bins=20, edgecolor='black', color='maroon')
    plt.title('Хистограм промене цена акције Peugeot компаније')
    plt.xlabel('Дневна промена цене')
    plt.ylabel('Учестаност')
    plt.savefig("gen/promena_cene.png")
    plt.close()

def plot_candlestick_daytoday(df):
    sd = pd.Timestamp(2019,1,1)
    ed = pd.Timestamp(2019,1,31)
    df_month = df[(df['Date'] >= sd) & (df['Date'] <= ed)]

    #ohlc = df[['Date', 'Open', 'Daily_High', 'Daily_Low', 'Closing_Price']].set_index('Date')
    ohlc = df_month[['Date', 'Open', 'Daily_High', 'Daily_Low', 'Closing_Price']].set_index('Date')
    ohlc.columns = ['Open', 'High', 'Low', 'Close']
    mpf.plot(ohlc, type='candle', style='yahoo', title='Акције Peugeot компаније', ylabel='Цена акције', show_nontrading=False,savefig="gen/marketplot_jan.png")

def plot_volume(df):
    temp = df
    df['Volume_Quartiles'] = pd.qcut(df['Volume'], q=[0, 0.25, 0.5, 0.75, 1.0], labels=['Q1', 'Q2', 'Q3', 'Q4'])
    volume_quartile_counts = df['Volume_Quartiles'].value_counts()
    quartile_ranges = df.groupby('Volume_Quartiles')['Volume'].agg(['min', 'max'])
    plt.figure(figsize=(9, 6))
    volume_quartile_counts.sort_index().plot(kind='bar', color='maroon', width=0.8)
    
    bars = plt.bar(volume_quartile_counts.index, volume_quartile_counts.sort_index(), color='maroon', width=0.8)
    
    for bar, (q, (min_val, max_val)) in zip(bars, quartile_ranges.iterrows()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'Распон: {min_val:.2f}-{max_val:.2f}', ha='center', va='bottom')
    plt.title('Обими трговања акцијама по квартилима')
    plt.xlabel('Квартил Обима')
    plt.ylabel('Број трансакција')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("gen/volume_quartiles_plot.png")
    plt.close()

def plot_corr(df):
    cm = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='rocket', fmt=".2f", linewidths=0.5)
    plt.title('Матрица корелације')
    plt.savefig("gen/cm.png")
    plt.close()


df = load_data(dt.datetime(2015,1,1),dt.datetime(2020,1,1))
#name_counts = df.groupby(['Name']).size()

#plot_komp_cnt(name_counts)


#print(df)
#plot_closing_price(df)
#plot_daily_price_changes(df)
#plot_candlestick_daytoday(df)
#plot_volume(df)
#tmp = df.drop(columns=["Name"])
#plot_corr(df.drop(columns=["Name"]))
#plot_ltsm_model("best_weights.h5")


