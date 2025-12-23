#!/bin/bash

echo "🚀 CLOUD COST ANOMALY DETECTION - HACKATHON DEMO"
echo "=================================================="
echo ""

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, fastapi, sklearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed"
else
    echo "⚠️  Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "🔍 Running system verification..."
python3 verify_system.py

echo ""
echo "🎯 DEMO OPTIONS:"
echo "1. Interactive Dashboard: streamlit run dashboard/app.py"
echo "2. API Service: uvicorn api.main:app --reload"
echo "3. Docker Deploy: docker build -t cost-anomaly . && docker run -p 8000:8000 cost-anomaly"
echo ""
echo "📊 Dashboard URL: http://localhost:8501"
echo "🔗 API Docs URL: http://localhost:8000/docs"
echo ""
echo "🏆 Ready for hackathon presentation!"