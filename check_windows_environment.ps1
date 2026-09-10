#requires -Version 5.1

& {
    function Section {
        param([string]$Title)
        Write-Output ""
        Write-Output ("== " + $Title + " ==")
    }

    function Find-App {
        param([string]$Name)
        return Get-Command $Name -CommandType Application `
            -ErrorAction SilentlyContinue | Select-Object -First 1
    }

    Write-Output "DL2026 Windows environment preflight"
    Write-Output "Read-only: no download, install, OS change, driver change, or policy change."

    Section "Windows and hardware"
    $osText = [Environment]::OSVersion.VersionString
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $osText = "{0}; version {1}; build {2}" -f `
            $os.Caption, $os.Version, $os.BuildNumber
    }
    catch {
        Write-Output "[INFO] Detailed Windows information is unavailable."
    }

    $osArch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) {
        $osArch = $env:PROCESSOR_ARCHITEW6432
    }

    Write-Output ("OS: " + $osText)
    Write-Output ("OS architecture: " + $osArch)
    Write-Output ("64-bit OS: " + [Environment]::Is64BitOperatingSystem)
    Write-Output ("64-bit PowerShell process: " + [Environment]::Is64BitProcess)
    Write-Output ("PowerShell: " + $PSVersionTable.PSVersion)

    try {
        $computer = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
        $memoryGb = [Math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
        Write-Output ("Memory: {0} GB" -f $memoryGb)
    }
    catch {
        Write-Output "[INFO] Memory information is unavailable."
    }

    try {
        $cpu = Get-CimInstance Win32_Processor -ErrorAction Stop |
            Select-Object -First 1
        Write-Output ("CPU: " + $cpu.Name.Trim())
        Write-Output ("CPU cores/logical processors: {0}/{1}" -f `
            $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors)
    }
    catch {
        Write-Output "[INFO] CPU information is unavailable."
    }

    Section "Display adapters"
    $videoNames = @()
    try {
        $videoNames = @(
            Get-CimInstance Win32_VideoController -ErrorAction Stop |
            ForEach-Object { $_.Name }
        )
        if ($videoNames.Count -eq 0) {
            Write-Output "[INFO] No display adapter was reported."
        }
        else {
            foreach ($name in $videoNames) {
                Write-Output ("Adapter: " + $name)
            }
        }
    }
    catch {
        Write-Output "[INFO] Display-adapter query is unavailable."
    }

    Section "Conda"
    $conda = Get-Command conda -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $condaInvoke = $null
    $condaLocation = $null

    if ($env:CONDA_EXE -and
        (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
        $condaInvoke = $env:CONDA_EXE
        $condaLocation = $env:CONDA_EXE
    }
    elseif ($null -ne $conda) {
        $condaInvoke = "conda"
        $condaLocation = $env:CONDA_EXE
        if (-not $condaLocation -and $conda.Path) {
            $condaLocation = $conda.Path
        }
        if (-not $condaLocation) {
            $condaLocation = [string]$conda.CommandType
        }
    }
    else {
        $condaCandidates = @(
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
            "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
            "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
            "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
            "$env:ProgramData\miniconda3\Scripts\conda.exe",
            "$env:ProgramData\anaconda3\Scripts\conda.exe"
        )
        foreach ($candidate in $condaCandidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $condaInvoke = $candidate
                $condaLocation = $candidate
                break
            }
        }
    }

    $condaReady = $false
    if ($condaInvoke) {
        try {
            $condaVersion = @(& $condaInvoke --version 2>&1)
            if ($LASTEXITCODE -eq 0 -and $condaVersion.Count -gt 0) {
                $condaReady = $true
                Write-Output ("[OK] " + [string]$condaVersion[0])
                Write-Output ("Conda location: " + $condaLocation)
                if ($null -eq $conda -and -not $env:CONDA_DEFAULT_ENV) {
                    Write-Output "[INFO] Conda is installed but not initialized in this shell."
                }
            }
            else {
                Write-Output "[NOT READY] Conda was found but could not run."
            }
        }
        catch {
            Write-Output ("[NOT READY] Conda could not run: " + $_.Exception.Message)
        }
    }
    else {
        Write-Output "[MISSING] Conda was not found in the shell or common install paths."
    }

    if ($env:CONDA_DEFAULT_ENV) {
        Write-Output ("Active Conda environment: " + $env:CONDA_DEFAULT_ENV)
        Write-Output ("Conda prefix: " + $env:CONDA_PREFIX)
    }
    else {
        Write-Output "[INFO] No active Conda environment."
    }

    Section "Python, pip, and torch"
    $pythonExe = $null
    $pythonPrefix = @()

    if ($env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $pythonExe = $candidate
        }
    }

    if (-not $pythonExe) {
        $python = Find-App "python.exe"
        if ($null -ne $python) {
            if ($python.Path -match "\\WindowsApps\\python(3)?\.exe$") {
                Write-Output ("[INFO] Microsoft Store Python alias found: " + $python.Path)
                Write-Output "The alias will not be launched."
            }
            else {
                $pythonExe = $python.Path
            }
        }
    }

    if (-not $pythonExe) {
        $py = Find-App "py.exe"
        if ($null -ne $py) {
            $pythonExe = $py.Path
            $pythonPrefix = @("-3")
            Write-Output ("[INFO] Using the Python launcher for inspection: " + $pythonExe)
        }
    }

    $torchState = "unknown"
    if ($pythonExe) {
        $probe = @'
import sys
print('Python version: {}'.format(sys.version.split()[0]))
print('Python executable: {}'.format(sys.executable))
print('Python 64-bit: {}'.format(sys.maxsize > 2**32))
try:
    import torch
except Exception as exc:
    print('[NOT READY] torch import: {}: {}'.format(type(exc).__name__, exc))
    print('DL2026_TORCH_STATE=missing-or-broken')
else:
    print('torch version: {}'.format(torch.__version__))
    print('torch build CUDA: {}'.format(torch.version.cuda or 'None (CPU build)'))
    available = torch.cuda.is_available()
    print('torch.cuda.is_available: {}'.format(available))
    print('torch GPU count: {}'.format(torch.cuda.device_count()))
    if available:
        for index in range(torch.cuda.device_count()):
            print('torch GPU {}: {}'.format(index, torch.cuda.get_device_name(index)))
        state = 'cuda-ready'
    elif torch.version.cuda:
        state = 'cuda-build-not-ready'
    else:
        state = 'cpu-build'
    print('DL2026_TORCH_STATE=' + state)
'@
        $probeArgs = @()
        $probeArgs += $pythonPrefix
        $probeArgs += @("-X", "utf8", "-c", $probe)

        try {
            $probeOutput = @(& $pythonExe @probeArgs 2>&1)
            foreach ($line in $probeOutput) {
                $text = [string]$line
                if ($text -like "DL2026_TORCH_STATE=*") {
                    $torchState = $text.Substring("DL2026_TORCH_STATE=".Length)
                }
                else {
                    Write-Output $text
                }
            }
            $pipArgs = @()
            $pipArgs += $pythonPrefix
            $pipArgs += @("-m", "pip", "--version")
            $pipOutput = @(& $pythonExe @pipArgs 2>&1)
            if ($LASTEXITCODE -eq 0 -and $pipOutput.Count -gt 0) {
                Write-Output ("[OK] " + [string]$pipOutput[0])
            }
            else {
                Write-Output "[NOT READY] pip could not run in this Python."
            }
        }
        catch {
            Write-Output ("[NOT READY] Python could not run: " + $_.Exception.Message)
        }
    }
    else {
        Write-Output "[MISSING] A usable Python interpreter was not found."
    }

    Section "NVIDIA"
    $smi = Find-App "nvidia-smi.exe"
    $smiPath = $null
    if ($null -ne $smi) {
        $smiPath = $smi.Path
    }
    else {
        $smiCandidates = @(
            "$env:WINDIR\System32\nvidia-smi.exe",
            "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        )
        foreach ($candidate in $smiCandidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $smiPath = $candidate
                break
            }
        }
    }

    $smiReady = $false
    if ($smiPath) {
        $smiSummary = @(& $smiPath 2>&1)
        $smiSummaryText = $smiSummary -join "`n"
        if ($LASTEXITCODE -eq 0) {
            $smiRows = @(
                & $smiPath `
                    --query-gpu=index,name,driver_version,memory.total `
                    --format=csv,noheader 2>$null
            )
            if ($LASTEXITCODE -eq 0 -and $smiRows.Count -gt 0) {
                $smiReady = $true
                Write-Output "index, name, driver, total memory"
                foreach ($row in $smiRows) {
                    Write-Output ([string]$row)
                }
                if ($smiSummaryText -match "CUDA(?: UMD)? Version:\s*([0-9.]+)") {
                    Write-Output ("Driver CUDA ceiling: " + $Matches[1])
                }
            }
        }
        if (-not $smiReady) {
            Write-Output "[NOT READY] nvidia-smi exists but its GPU query failed."
        }
        Write-Output ("nvidia-smi path: " + $smiPath)
    }
    else {
        Write-Output "[INFO] nvidia-smi was not found."
    }

    Section "Developer tools"
    $git = Find-App "git.exe"
    if ($null -ne $git) {
        $gitVersion = @(& $git.Path --version 2>&1)
        if ($gitVersion.Count -gt 0) {
            Write-Output ("[OK] " + [string]$gitVersion[0])
        }
        Write-Output ("Git path: " + $git.Path)
    }
    else {
        Write-Output "[MISSING] Git command was not found. ZIP download is an alternative."
    }

    $code = Find-App "code.cmd"
    if ($null -eq $code) {
        $code = Find-App "code.exe"
    }
    if ($null -ne $code) {
        $codeVersion = @(& $code.Path --version 2>&1)
        if ($codeVersion.Count -gt 0) {
            Write-Output ("[OK] VS Code CLI " + [string]$codeVersion[0])
        }
        Write-Output ("VS Code CLI path: " + $code.Path)
    }
    else {
        Write-Output "[INFO] VS Code CLI was not found. VS Code may still be installed."
    }

    $hasNvidiaAdapter = @(
        $videoNames | Where-Object { $_ -match "NVIDIA" }
    ).Count -gt 0
    $hasOtherGpu = @(
        $videoNames | Where-Object { $_ -match "AMD|Radeon|Intel|Arc" }
    ).Count -gt 0

    Section "Manual next step"
    if (-not $condaReady) {
        Write-Output "1. Install or open 64-bit Miniconda, then rerun this preflight."
    }
    elseif (-not $env:CONDA_DEFAULT_ENV) {
        Write-Output "1. Create and activate dl2026, then rerun this preflight."
    }
    elseif ($env:CONDA_DEFAULT_ENV -eq "base") {
        Write-Output "1. Do not install course packages in base; activate dl2026 first."
    }
    elseif ($env:CONDA_DEFAULT_ENV -ne "dl2026") {
        Write-Output ("1. Current environment is " + $env:CONDA_DEFAULT_ENV +
            "; create or activate dl2026 before course installation.")
    }
    else {
        Write-Output ("1. Active Conda environment: " + $env:CONDA_DEFAULT_ENV)
    }

    if ($torchState -eq "cuda-ready") {
        Write-Output "2. Current PyTorch CUDA is ready. Do not reinstall it."
    }
    elseif ($torchState -eq "cuda-build-not-ready") {
        Write-Output "2. A CUDA build of PyTorch exists, but CUDA is not usable."
        Write-Output "   Check the active environment and NVIDIA driver; do not add a Toolkit at random."
    }
    elseif ($smiReady) {
        Write-Output "2. NVIDIA GPU and driver are visible: use the NVIDIA route in the guide."
        if ($torchState -eq "cpu-build") {
            Write-Output "   Current PyTorch is CPU-only. Use a clean environment for the GPU route."
        }
    }
    elseif ($hasNvidiaAdapter -or $smiPath) {
        Write-Output "2. NVIDIA hardware may exist, but its driver is not ready."
        Write-Output "   Use CPU for class or repair the official NVIDIA driver first."
    }
    elseif ($torchState -eq "cpu-build" -and $hasOtherGpu) {
        Write-Output "2. Current CPU build of PyTorch is ready. Do not reinstall it."
    }
    elseif ($hasOtherGpu) {
        Write-Output "2. Only non-NVIDIA graphics were detected: use the CPU route for this course."
    }
    else {
        Write-Output "2. GPU type is unknown: the CPU route is the safest choice."
    }

    if ($torchState -eq "missing-or-broken" -or $torchState -eq "unknown" -or `
        ($torchState -eq "cpu-build" -and $smiReady)) {
        Write-Output ""
        Write-Output "Run exactly one command manually after dl2026 is active:"
        Write-Output "CPU:    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu"
        Write-Output "CUDA126: python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126"
        Write-Output "CUDA128: python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128"
    }

    Write-Output "3. Read the Windows setup guide linked from README.md."
    Write-Output "4. After manual installation: python check_environment.py"
    Write-Output "5. Final smoke check: python test_device_speed_demo.py"
    Write-Output ""
    Write-Output "Preflight finished. Nothing was installed or changed."
}
