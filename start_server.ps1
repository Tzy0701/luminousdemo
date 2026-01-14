# Luminous Fingerprint Analysis - Server Startup Script
# This script starts both the API backend and Streamlit frontend
# configured for network access from other laptops

# ============================================================================
# CONFIGURATION - CHANGE THIS TO YOUR SERVER'S IP ADDRESS
# ============================================================================
# To find your IP: Run 'ipconfig' and look for IPv4 Address
$SERVER_IP = "161.142.146.213"  # Your actual server IP

# API Port (default: 8000)
$API_PORT = "8000"

# Streamlit Port (default: 8501)
$STREAMLIT_PORT = "8501"

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  LUMINOUS FINGERPRINT ANALYSIS SYSTEM - SERVER MODE" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Server IP:        $SERVER_IP" -ForegroundColor White
Write-Host "  API Port:         $API_PORT" -ForegroundColor White
Write-Host "  Streamlit Port:   $STREAMLIT_PORT" -ForegroundColor White
Write-Host ""

# Set environment variables for Streamlit to find the API
$env:API_HOST = $SERVER_IP
$env:API_PORT = $API_PORT

Write-Host "Starting Backend API (FastAPI)..." -ForegroundColor Yellow

# Start API in a new window
$apiProcess = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$PWD'; Write-Host 'Backend API Starting...' -ForegroundColor Green; python api.py"
) -PassThru -WindowStyle Normal

Write-Host "  API process started (PID: $($apiProcess.Id))" -ForegroundColor Green
Write-Host "  Waiting 10 seconds for API to initialize..." -ForegroundColor White

# Wait for API to start
Start-Sleep -Seconds 10

# Test if API is responding
Write-Host "  Testing API connection..." -ForegroundColor White
try {
    $response = Invoke-WebRequest -Uri "http://${SERVER_IP}:${API_PORT}/" -TimeoutSec 5 -UseBasicParsing
    Write-Host "  ✓ API is responding!" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ API not responding yet (this is normal, it may still be starting)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting Streamlit Frontend..." -ForegroundColor Yellow

# Start Streamlit in current window
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  SYSTEM READY" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the application:" -ForegroundColor Yellow
Write-Host "  From this computer:    http://localhost:$STREAMLIT_PORT" -ForegroundColor White
Write-Host "  From other computers:  http://${SERVER_IP}:${STREAMLIT_PORT}" -ForegroundColor White
Write-Host ""
Write-Host "API Endpoint:" -ForegroundColor Yellow
Write-Host "  Backend URL:           http://${SERVER_IP}:${API_PORT}" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Run Streamlit (this will block until user stops it)
streamlit run streamlit_app.py --server.port $STREAMLIT_PORT --server.address "0.0.0.0"

# Cleanup when Streamlit stops
Write-Host ""
Write-Host "Shutting down..." -ForegroundColor Yellow
Write-Host "Stopping API process..." -ForegroundColor White

# Stop the API process
if ($apiProcess -and !$apiProcess.HasExited) {
    Stop-Process -Id $apiProcess.Id -Force
    Write-Host "  ✓ API stopped" -ForegroundColor Green
}

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Green

