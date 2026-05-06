# Setting up a VNC host

`daedalus` controls a target machine over VNC. You install a small VNC **server** on the host you want to control, and the agent connects from the control plane.

## Windows (recommended for Phase 0)

We recommend [TightVNC](https://www.tightvnc.com/) or [UltraVNC](https://uvnc.com/) — both are free and widely used.

1. Download the TightVNC installer.
2. During install, choose **Server only** (you don't need the viewer on the controlled machine).
3. When prompted, set:
   - **Primary password** (for full control) — use a strong random string. The agent will pass this to the backend.
   - **Listen port**: `5900` (default).
   - **Allow loopback connections only**: leave unchecked if connecting from another machine.
4. Open Windows Firewall for port `5900/tcp` from the control plane's IP.
5. Lock the screen orientation to landscape and set resolution to `1920x1080` (the Phase 0 baseline).

Verify from the control plane:

```bash
# Test from another machine on the same network
vncdo -s WINDOWS_HOST_IP::5900 -p YOUR_PASSWORD capture screen.png
```

## macOS

Built-in. Enable **System Settings → General → Sharing → Screen Sharing**, then set a VNC password under "Computer Settings…".

## Linux

```bash
sudo apt install tigervnc-standalone-server
vncpasswd                          # set a password
tigervncserver -localhost no -geometry 1920x1080 :1
```

This will listen on port `5901` (display `:1`).

## Security notes

- VNC traffic is not encrypted by default. Tunnel over SSH for anything beyond a closed lab network:
  ```bash
  ssh -L 5900:localhost:5900 user@host
  # then point daedalus at --host localhost --port 5900
  ```
- Use a unique password per host. Store it in a credentials manager, not in `config.yaml`.
- Never expose VNC to the public internet directly.
