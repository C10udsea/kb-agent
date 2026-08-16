# kb-agent Windows 一键初始化脚本
# 用法（在项目根目录）:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# 或者先允许本次脚本执行:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== kb-agent Windows 环境初始化 ===" -ForegroundColor Cyan

# 1. 选择 Python：优先 3.12 / 3.11 / 3.10（ChromaDB 兼容性最好）
$PythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($Ver in @("3.12", "3.11", "3.10")) {
        py "-$Ver" -c "import sys" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $PythonCmd = @("py", "-$Ver")
            Write-Host "使用 Python $Ver" -ForegroundColor Green
            break
        }
    }
}
if (-not $PythonCmd) {
    $PythonCmd = @("python")
    python --version
}

# 2. 创建 Windows 虚拟环境（Linux 创建的 .venv 在 Windows 上不可用）
if (Test-Path .venv) {
    Write-Host "删除旧的 .venv ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}
Write-Host "创建虚拟环境 .venv ..." -ForegroundColor Yellow
$PyLauncher = $PythonCmd[0]
if ($PythonCmd.Length -gt 1) {
    $PyVerArg = $PythonCmd[1]
    & $PyLauncher $PyVerArg -m venv .venv
} else {
    & $PyLauncher -m venv .venv
}
if ($LASTEXITCODE -ne 0) { throw "venv 创建失败" }

# 3. 安装依赖
Write-Host "安装依赖（首次运行需要几分钟）..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

# 4. 检查 .env
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "已创建 .env，请打开它填入 DeepSeek / GLM 的 Key！" -ForegroundColor Magenta
} else {
    Write-Host ".env 已存在。" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== 初始化完成 ===" -ForegroundColor Cyan
Write-Host "接下来手动执行："
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\ingest.py --clear"
Write-Host "  python scripts\chat.py -q `"什么是过拟合？`""
Write-Host "  python scripts\agent.py -q `"对比 RAG 和 Agent 的区别`""
