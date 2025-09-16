#!/usr/bin/env python3
"""
Simple HTTP server for SkyReels V2 Web UI preview on port 3000
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 3000

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=Path(__file__).parent, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/preview.html'
        return super().do_GET()

def main():
    print("🎬 SkyReels V2 Web UI Server")
    print("=" * 40)
    print(f"🚀 Starting server on port {PORT}...")
    print(f"📱 Open your browser to: http://localhost:{PORT}")
    print("⚡ Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Change to the web_ui directory
    os.chdir(Path(__file__).parent)
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"✅ Server running at http://localhost:{PORT}")
            
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}')
                print("🌐 Opened browser automatically")
            except:
                print("💡 Please open your browser manually to http://localhost:3000")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use. Please try a different port or stop the existing service.")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()