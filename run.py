from source.data_processing import FeatureSelector
from source.app_dash import app, server

if __name__ == "__main__":
    app.run_server(host = '0.0.0.0', port = 8050, debug = True)