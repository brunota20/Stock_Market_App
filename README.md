# Stock Market App

This is a web application that provides insights into the stock market using the Streamlit library. It includes features such as market overview, monthly returns analysis, fundamental information, and even an index prediction tool.

## Getting Started

1. Install the required packages:

   ```bash
   pip install streamlit pandas yfinance seaborn matplotlib scikit-learn plotly fundamentus
2. Run the application:

   streamlit run app.py

## Features

- **Market Overview:** Provides an overview of various market indices and assets.
- **Monthly Returns:** Analyzes monthly returns for selected indices or stocks.
- **Fundamentals:** Displays fundamental information about selected stocks using the Fundamentus library.
- **Index Prediction:** Predicts future index prices using a RandomForestRegressor.

## Usage

1. Open your web browser and go to `http://localhost:8501` (Streamlit's default local host).

2. Use the sidebar to navigate between different features.

## Important Notes

- Stock price prediction is a complex and uncertain task due to various influencing factors.
- Make sure to keep your Python dependencies up-to-date using a virtual environment or a requirements file.

## Acknowledgements

- This project was created using the Streamlit library.
- Fundamental information is fetched using the Fundamentus library.
- Stock price data is collected from Yahoo Finance using the yfinance library.

## License

This project is licensed under the MIT License.

https://stockmarketapp-wrpqvipappcdkuxqmh9cawh.streamlit.app/
