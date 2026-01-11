#!/bin/bash
# Zerve deployment start script
uvicorn api.deploy:app --host 0.0.0.0 --port 8000