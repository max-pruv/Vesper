#!/bin/bash
# Vesper — VPS Security Hardening Script
# Run on your VPS: bash deploy/secure_vps.sh
set -e

echo "=========================================="
echo "  Vesper — VPS Security Hardening"
echo "=========================================="

# 1. Configure firewall (UFW)
echo "[1/4] Configuring firewall..."
apt-get install -y ufw > /dev/null 2>&1
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 8080/tcp  # Vesper dashboard
ufw --force enable
echo "  Firewall: only SSH (22) and dashboard (8080) open"

# 2. Harden SSH
echo "[2/4] Hardening SSH..."
SSHD_CONFIG="/etc/ssh/sshd_config"
# Disable root password login (use SSH keys instead)
if ! grep -q "^PermitRootLogin prohibit-password" "$SSHD_CONFIG"; then
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' "$SSHD_CONFIG"
fi
# Disable empty passwords
sed -i 's/^#*PermitEmptyPasswords.*/PermitEmptyPasswords no/' "$SSHD_CONFIG"
# Limit auth attempts
if ! grep -q "^MaxAuthTries" "$SSHD_CONFIG"; then
    echo "MaxAuthTries 3" >> "$SSHD_CONFIG"
fi
systemctl reload sshd 2>/dev/null || systemctl reload ssh 2>/dev/null || true
echo "  SSH: root password login disabled, max 3 auth attempts"

# 3. Install fail2ban
echo "[3/4] Installing fail2ban..."
apt-get install -y fail2ban > /dev/null 2>&1
cat > /etc/fail2ban/jail.local << 'JAILEOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
JAILEOF
systemctl enable fail2ban
systemctl restart fail2ban
echo "  fail2ban: bans IPs after 3 failed SSH attempts for 1 hour"

# 4. Auto security updates
echo "[4/4] Enabling automatic security updates..."
apt-get install -y unattended-upgrades > /dev/null 2>&1
dpkg-reconfigure -plow unattended-upgrades 2>/dev/null || true
echo "  Automatic security updates enabled"

echo ""
echo "=========================================="
echo "  Security hardening complete!"
echo "=========================================="
echo ""
echo "  Next steps for maximum security:"
echo "  1. Set up SSH keys: ssh-copy-id root@your-vps"
echo "  2. Enable 2FA in Vesper dashboard"
echo "  3. Use a strong DASHBOARD_PASSWORD"
echo "=========================================="
