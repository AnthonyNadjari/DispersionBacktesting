#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import scrolledtext
import PIL
from PIL import ImageTk
from PIL import Image


def get_params():

    global list_tickers,strike,start_date,N,weights,index_ticker
    weights =[]
    list_tickers= []
    strike = 0
    start_date =0
    N = 0
    index_ticker = 0
    window = tk.Tk()

    window.geometry("800x600")
    window.title("Dispersion Backtester - Input Parameters")
    
    text_stocks = scrolledtext.ScrolledText(window, width =12, height = 10)
    text_stocks.place(x=50,y=210)
    label_stocks = tk.Label(text = "Basket Stocks tickers")
    label_stocks.place(x=50, y=190)


    text_weights = scrolledtext.ScrolledText(window, width =6, height = 10)
    text_weights.place(x=230,y=210)
    label_weights = tk.Label(text = "Basket Stocks weights")
    label_weights.place(x=200, y=190)


    label_index = tk.Label(text = "Index Ticker")
    label_index.place(x=420,y=190)
    text_index = tk.Entry()
    text_index.place(x=420,y=210)

    label_strike = tk.Label(text = "Strike level")
    label_strike.place(x= 420, y=240)
    text_strike = tk.Entry(width =5)
    text_strike.place(x=420, y=260)
    window["bg"]="#00b0f0"



    label_start_date = tk.Label(text ="Backtest Start Date")
    label_start_date.place(x=420,y=290)
    text_start_date = tk.Entry(width = 10)
    text_start_date.place(x=420,y=310)


    label_N = tk.Label(text = "Number of Periods (N)")
    label_N.place(x=420,y=340)
    text_N = tk.Entry(width = 4)
    text_N.place(x=420,y=360)

    img = ImageTk.PhotoImage(Image.open("C:/Users/nadja/OneDrive/Bureau/code/backtest_dispersion/barlcays_logo.png").resize((300,170)))

    label_photo=tk.Label(window,image=img)
    label_photo.place(x=100,y=50)
    label_photo.pack()
    
    
    window.iconbitmap(False,"C:/Users/nadja/OneDrive/Bureau/code/backtest_dispersion/barclays-eagle-logo.ico")
    def get_button():
        global list_tickers,strike,start_date,N,weights,index_ticker
        list_tickers.append(text_stocks.get("1.0",'end-1c'))
        strike = text_strike.get()
        index_ticker = text_index.get()
        N = text_N.get()
        start_date = text_start_date.get()

        window.destroy()


    button1 = tk.Button(text="Run Backtest", command=get_button, width = 40, height = 4)
    button1.place(x=270,y=450)
    window.mainloop() 
get_params()

