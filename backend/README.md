# ClockScan Backend — Deployment Guide

## What's in this folder

| File | Purpose |
|------|---------|
| `clock_reader.py` | Core OpenCV clock detection algorithm |
| `main.py` | FastAPI REST server |
| `requirements.txt` | Python dependencies (~50MB total) |
| `Dockerfile` | For Docker deployment |
| `clockscan.service` | For systemd (no Docker) |

## Resource usage
- RAM: ~120MB at rest, ~200MB peak
- CPU: <100ms per image on 1 vCPU
- Disk: ~80MB (OpenCV + deps)

---

## Option A — Deploy directly on Linux server (recommended for your specs)

### 1. Upload files to your server
```bash
scp -r clockscan-backend/ user@YOUR_SERVER_IP:~/
```

### 2. SSH into your server
```bash
ssh user@YOUR_SERVER_IP
```

### 3. Install Python + create virtualenv
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip libglib2.0-0
cd ~/clockscan-backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Test it runs
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# Visit http://YOUR_SERVER_IP:8000 — should return {"status":"ok"}
# Ctrl+C to stop
```

### 5. Install as a system service (runs forever, auto-restarts)
```bash
# Edit clockscan.service — change User=ubuntu to your actual username
nano clockscan.service

sudo cp clockscan.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable clockscan
sudo systemctl start clockscan

# Check it's running
sudo systemctl status clockscan
```

### 6. Open firewall port
```bash
sudo ufw allow 8000
# Or if using iptables:
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

---

## Option B — Docker (if Docker is installed)

```bash
docker build -t clockscan .
docker run -d --name clockscan -p 8000:8000 --restart always clockscan
```

---

## Test the API

```bash
# Health check
curl http://YOUR_SERVER_IP:8000/health

# Test with a clock image
curl -X POST http://YOUR_SERVER_IP:8000/read-clock \
  -F "image=@/path/to/clock.jpg"
```

Expected response:
```json
{
  "time": "10:10",
  "period": "AM/PM unknown (24h not determinable from analog)",
  "confidence": "HIGH",
  "notes": "Hour hand: 301.5°, Minute hand: 60.0°",
  "hour": 10,
  "minute": 10
}
```

---

## Tips for best accuracy

- **Good lighting** — avoid glare on glass clock faces
- **Full clock visible** — don't crop the edges
- **Straight-on angle** — avoid extreme side angles
- **Works best on** — wall clocks, simple dials, white faces with dark hands
- **Works poorly on** — ornate clocks, very small watches, blurry images

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: libGL.so` | `sudo apt install libglib2.0-0` |
| Port 8000 refused | Check `sudo ufw status` and open port |
| Low confidence results | Ensure full clock face is visible and well lit |
| Service won't start | Check `journalctl -u clockscan -n 50` |
