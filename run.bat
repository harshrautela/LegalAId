@echo off
cd /d C:\Users\Siddharth Khatri\Documents\LegalAId
call conda activate legalaid
:loop
echo Starting Streamlit...
streamlit run ui/app.py --server.port 8501
echo Streamlit stopped. Restarting in 3 seconds...
timeout /t 3
goto loop