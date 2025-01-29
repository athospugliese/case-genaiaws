import uvicorn
from dotenv import load_dotenv
import multiprocessing
import subprocess
import os
import time

from core import settings

load_dotenv()

def run_uvicorn():
    """run the fastapi/uvicorn server"""
    uvicorn.run(
        "service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev()
    )

def run_streamlit():
    """run the streamlit app"""
    # adjust the path to your streamlit file
    streamlit_script = os.path.join(os.path.dirname(__file__), 'app.py')
    subprocess.run(["streamlit", "run", streamlit_script])

if __name__ == "__main__":
    # create separate processes for each service
    uvicorn_process = multiprocessing.Process(target=run_uvicorn)
    streamlit_process = multiprocessing.Process(target=run_streamlit)

    try:
        # start both services
        uvicorn_process.start()
        # small delay to avoid race conditions
        time.sleep(2)
        streamlit_process.start()

        # keep processes alive
        uvicorn_process.join()
        streamlit_process.join()
        
    except KeyboardInterrupt:
        # clean up if user stops with ctrl+c
        uvicorn_process.terminate()
        streamlit_process.terminate()
        print("\nservices stopped")