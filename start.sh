#!/usr/bin/env bash
APP_NAME=${APP_NAME:-app_predictor.py}
streamlit run "$APP_NAME" --server.port ${PORT:-8501} --server.address 0.0.0.0
