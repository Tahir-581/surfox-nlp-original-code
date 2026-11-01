# Run this script as Administrator to allow LAN access to Surfox.
# Usage: Right-click PowerShell -> Run as administrator, then:
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   & "D:\Tahir\surfox-nlp-scrapling\scripts\open_lan_access.ps1"

$ErrorActionPreference = "Stop"

Write-Host "Setting active network profile to Private (allows device-to-device access)..."
Get-NetConnectionProfile | ForEach-Object {
    Set-NetConnectionProfile -InterfaceIndex $_.InterfaceIndex -NetworkCategory Private
}

$rules = @(
    @{ Name = "Surfox Backend 8010"; Port = "8010" },
    @{ Name = "Surfox Frontend 3010"; Port = "3010-3020" }
)

foreach ($rule in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Firewall rule already exists: $($rule.Name)"
        continue
    }
    Write-Host "Adding firewall rule: $($rule.Name) (TCP $($rule.Port))"
    New-NetFirewallRule `
        -DisplayName $rule.Name `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort $rule.Port `
        -Profile Any | Out-Null
}

Write-Host ""
Write-Host "Done. Restart run_services.py, then open from another device:"
Write-Host "  LAN mode (recommended):  http://<your-ip>:8010"
Write-Host "  Dev mode:                  http://<your-ip>:3010"
Write-Host ""
Get-NetConnectionProfile | Select-Object Name, NetworkCategory | Format-Table -AutoSize
