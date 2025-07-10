Project Structure

Project-6_Stock_price_predictor/
├── stock_predict_lstm.py # Main script to train/predict using LSTM
├── app.py # Streamlit web app
├── lstm_stock_model.h5 # Saved model file
├── requirements.txt # Python package dependencies
└── README.md # Project documentation

How to run

1. Clone the repo

2. Create a virtual environment
   python -m venv .venv
   .venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run the LSTM model script
   python stock_predict_lstm.py
   Trains the model if not already saved, and generates a price prediction chart.

5. Launch the Web App
   streamlit run app.py
   Go to http://localhost:8501 in your browser.
