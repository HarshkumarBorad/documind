# Aurora Desktop — Installation Guide

Aurora Desktop is a cross-platform client for managing Aurora compute jobs and storage. This guide covers installation on Windows, macOS, and Linux.

## System requirements

- Operating system: Windows 10 (1809+), macOS 11+, Ubuntu 22.04+ or equivalent.
- Memory: 4 GB RAM minimum, 8 GB recommended.
- Disk: 500 MB free space.
- Network: HTTPS access to `api.aurora.example` (port 443).

## Windows installation

1. Download `AuroraDesktop-Setup.exe` from `downloads.aurora.example`.
2. Right-click the installer and select **Run as administrator**.
3. Follow the prompts. Accept the default install location (`C:\Program Files\Aurora`).
4. After installation, launch Aurora Desktop from the Start menu.
5. Sign in with your Aurora account.

## macOS installation

1. Download `AuroraDesktop.dmg` from `downloads.aurora.example`.
2. Double-click the `.dmg` to mount it.
3. Drag the Aurora Desktop icon into the Applications folder.
4. Launch from Applications. macOS may prompt about an unidentified developer — open **System Settings → Privacy & Security** and click "Open Anyway".

## Linux installation

We ship a `.deb` package for Debian/Ubuntu and a `.rpm` for Fedora/RHEL.

```bash
# Debian / Ubuntu
sudo dpkg -i aurora-desktop_1.4.0_amd64.deb

# Fedora / RHEL
sudo rpm -i aurora-desktop-1.4.0.x86_64.rpm
```

For other distributions, a portable tarball is available from the downloads page.

## Configuration

On first launch, Aurora Desktop will prompt for:

1. Your Aurora API key (or interactive OAuth login).
2. The default project to use.
3. Telemetry preferences. You can disable telemetry at any time in **Settings → Privacy**.

The configuration file is stored at:

- Windows: `%APPDATA%\Aurora\config.json`
- macOS: `~/Library/Application Support/Aurora/config.json`
- Linux: `~/.config/aurora/config.json`

## Troubleshooting

**"Cannot connect to api.aurora.example"** — check your firewall and ensure HTTPS to `api.aurora.example:443` is allowed. Corporate proxies often need configuration in **Settings → Network**.

**"API key invalid"** — generate a new key from the web console. Old keys are revoked when rotated; the desktop client does not auto-pick-up rotated keys.

**Slow startup** — clear the local cache via **Settings → Advanced → Clear cache**. Aurora Desktop will redownload metadata on next launch.

## Uninstall

- Windows: **Settings → Apps → Aurora Desktop → Uninstall**.
- macOS: drag the app from Applications to Trash; remove `~/Library/Application Support/Aurora` for a clean wipe.
- Linux: `sudo apt remove aurora-desktop` or equivalent for your distribution.
