# Analysis of COVID-19 Impact On (BEACH)  Stocks  and Price Prediction

## Introduction:
We realize that the [COVID-19](https://en.wikipedia.org/wiki/Coronavirus_disease_2019) has flipped around the economy. Companies in the travel and entertainment sectors, so-called BEACH stocks- were the absolute hardest hit. These stocks have seen more than $332 billion in esteem vanish during the months of Feb-Mar 2020 [source](https://markets.businessinsider.com/news/stocks/covid-19-downturn-beach-stocks-1029032907#)<br>
On the other hand, Time series forecasting, especially stock prediction, has been a hot topic for decades. Obviously, predicting the stock market is one of the most challenging things to do, and so many smart people and organizations are involved in this area. There are many variables that will [affect the price](https://www.udacity.com/course/machine-learning-for-trading--ud501): the earnings of a company, the supply and demand at that time, the trends of the overall economy, the political climate, pandemic and so on. 

## Objective:
Part of Udacity Data Scientist Nano degree program, I’ve selected Investment and Trading as Capstone Project, the objectives of this project are:<br>
1-	To quantify, compare, and visualize the impact of COVID-19 on the US stock market in the travel and entertainment sectors. The data stocks are considered from 2 Jan of(2019 and 2020) to May 31 (2019 and 2020), I've slected this period becuse the lockdown started in USA on March 19 2020 and As of 12 April, nearly 300 million people, or about 90 per cent of the population, are under some form of lockdown in the United States [source](https://www.businessinsider.com/us-map-stay-at-home-orders-lockdowns-2020-3)<br>
2-	To build a stock price predictor that takes daily trading data over a certain date range as input for selected stock simples(in travel and entertainment sectors) , and outputs projected estimates for given query dates. The system only will predict the Adjusted Close price.

## Data:
Data has been accessed from :
- Yahoo Finance using pandas_datareader library.
- USA COVID-19 from https://raw.githubusercontent.com

## Notebook and Report
In the repository, you can find: 
- The notebook, which To quantify, compare, and visualize the impact of COVID-19 on the US stock market in the travel and entertainment sector, in addtion it demonstrate how to predict stock price step by step with LSTM network and a Simple Moving Average model was provided to compare the performance.<br>
- you will find the asset folder under the master repo, i cound't copy it to the main. it is important in order to execute the Dash app
- you can find the detailed report for the work as pdf document.
- link for the final report on Medium  https://medium.com/@aliah.ghannam/analysis-of-covid-19-impact-on-beach-stocks-and-price-prediction-6c78aa1c00f3

## Prerequisites
- Pandas and Numpy: for data manipulation
- Matplotlib: for data visualisation
- Keras: for building and training the model
- sklearn: for data scaling
- Dash Plotly: for interactive web application creation

## Dash Web Application
In addition to the LSTM stock prediction model building and analysis, an interactive web application was also created. In order to run the web application, please follow these steps:<br>
- Git clone the repository https://github.com/alyahali/Udacity-Capstone-Project.git <br>
- Create the conda env with the requirement.yml <br> 
   - cd Capstone-Project-Stock-Predictor <br> 
   - conda env create -f environment.yml <br> 
   - conda activate ML <br>
- Run the web application by:<br>
  - cd lstm <br>
  - python index.py <br>
  - When the web server is on, go to http://127.0.0.1:8080/ Choose the stock symbol from the dropdown lists (selected stocks are provided.) Choose the start_date and end_date, and the click Run Model button. The  whole model running process will take about 3-8 minutes, you can see the progress in the terminal. After that, you can see the interactive plot for stock price (including train test and forcast).
  
  <img src="images/dash app.jpeg" width="80%" alt="Stock predictor dash app">
  <img src="images/dash app stock.jpeg" width="80%" alt="Stock predictor dash app">
 

## Acknowledgements

I wish to thank  [GitHub](https://github.com/) for the avilable examples,[raw.githubusercontent](https://raw.githubusercontent.com) for the data and thanks to [Udacity](https://www.udacity.com/) for 
advice and review.
