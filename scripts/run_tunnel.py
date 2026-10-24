"""Helper script to expose the local FastAPI recommendation server to a public HTTPS URL.

Supports Cloudflare Tunnel (cloudflared), localtunnel (npx), or ngrok.
Deliverable #1 for GeoAI Hackathon @ CSRE, IIT Bombay.
"""
import subprocess
import shutil
import sys
import time

PORT = 8000


def main():
    print("=" * 65)
    print("GeoAI Assistant: Public HTTP API Tunnel Launcher")
    print(f"Target local server: http://127.0.0.1:{PORT}")
    print("=" * 65)

    # Check for cloudflared
    if shutil.which("cloudflared"):
        print("\n[+] Found 'cloudflared'. Launching free Cloudflare Tunnel...")
        cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"]
        subprocess.run(cmd)
        return

    # Check for npx (localtunnel)
    if shutil.which("npx"):
        print("\n[+] Found 'npx'. Launching localtunnel...")
        cmd = ["npx", "localtunnel", "--port", str(PORT)]
        subprocess.run(cmd)
        return

    # Check for ngrok
    if shutil.which("ngrok"):
        print("\n[+] Found 'ngrok'. Launching ngrok tunnel...")
        cmd = ["ngrok", "http", str(PORT)]
        subprocess.run(cmd)
        return

    print("\n[-] No automatic tunneling CLI found (cloudflared, npx, or ngrok).")
    print("\nTo expose your server for judging:")
    print("Option A (Recommended): Download cloudflared from https://github.com/cloudflare/cloudflared/releases")
    print(f"  Run: cloudflared tunnel --url http://127.0.0.1:{PORT}")
    print("\nOption B: If Node.js is installed:")
    print(f"  Run: npx localtunnel --port {PORT}")
    print("\nOption C: If ngrok is installed:")
    print(f"  Run: ngrok http {PORT}")


if __name__ == "__main__":
    main()
