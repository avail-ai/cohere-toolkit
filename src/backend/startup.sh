#!/bin/sh

echo "Install additional dependencies"
pip install pysqlite3-binary
echo "Starting the backend server"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000