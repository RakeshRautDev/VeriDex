





import os
import sys
import subprocess
import time
import webbrowser
import requests
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}/"

def main():
    print("\n" + "="*50)
    print("🚀 Starting VeriDex Backend Server...")
    print(f"📍 Server URL: {URL}")
    print("="*50 + "\n")
    
    
    app_dir = Path(__file__).parent / "VeriDex_WebApp"
    original_dir = os.getcwd()
    os.chdir(app_dir)
    
    try:
        
        print("⏳ Starting Python server...")
        subprocess.Popen([sys.executable, "app.py"])
        
        
        print("⏳ Waiting for server to respond...")
        max_attempts = 60
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = requests.get(URL, timeout=2)
                print("✅ Server is ready!\n")
                break
            except:
                attempt += 1
                time.sleep(1)
        
        if attempt >= max_attempts:
            print("❌ Server failed to start. Check for errors above.")
            return
        
        
        print("🌐 Opening browser...")
        webbrowser.open(URL)
        print(f"✅ VeriDex opened at {URL}\n")
        print("="*50)
        print("💡 Server is running. Close the terminal to stop.")
        print("="*50 + "\n")
        
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping VeriDex...")
    
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
