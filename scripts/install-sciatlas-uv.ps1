param(
  [string]$RepoUrl = $(if ($env:SCIATLAS_REPO_URL) { $env:SCIATLAS_REPO_URL } else { "https://github.com/zjunlp/SciAtlas.git" }),
  [string]$Ref = $(if ($env:SCIATLAS_REF) { $env:SCIATLAS_REF } else { "" }),
  [string]$InstallDir = $(if ($env:SCIATLAS_INSTALL_DIR) { $env:SCIATLAS_INSTALL_DIR } else { Join-Path $HOME "SciAtlas" }),
  [switch]$SkipToolInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
  param([string]$Message)
  Write-Host ""
  Write-Host "==> $Message"
}

function Test-Command {
  param([string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Resolve-TargetPath {
  param([string]$PathValue)
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }
  return [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $PathValue))
}

function Ensure-Uv {
  if (Test-Command "uv") {
    return
  }

  Write-Step "Installing uv"
  $installer = Invoke-RestMethod "https://astral.sh/uv/install.ps1"
  Invoke-Expression $installer

  $candidatePaths = @(
    (Join-Path $HOME ".local\bin"),
    (Join-Path $HOME ".cargo\bin"),
    (Join-Path $env:USERPROFILE ".local\bin"),
    (Join-Path $env:USERPROFILE ".cargo\bin")
  ) | Where-Object { $_ -and (Test-Path $_) }

  foreach ($candidatePath in $candidatePaths) {
    if (($env:PATH -split [System.IO.Path]::PathSeparator) -notcontains $candidatePath) {
      $env:PATH = "$candidatePath$([System.IO.Path]::PathSeparator)$env:PATH"
    }
  }

  if (-not (Test-Command "uv")) {
    throw "uv was installed, but it is not on PATH yet. Open a new terminal and rerun this script."
  }
}

function Ensure-Git {
  if (Test-Command "git") {
    return
  }

  throw "git is required to download the full SciAtlas repository. Install Git, then rerun this script."
}

function Sync-Repository {
  param(
    [string]$TargetDir,
    [string]$RepositoryUrl,
    [string]$CheckoutRef
  )

  if (Test-Path $TargetDir) {
    $gitDir = Join-Path $TargetDir ".git"
    if (-not (Test-Path $gitDir)) {
      $children = @(Get-ChildItem -LiteralPath $TargetDir -Force)
      if ($children.Count -gt 0) {
        throw "Install directory exists and is not an empty Git checkout: $TargetDir"
      }
      Remove-Item -LiteralPath $TargetDir -Force
    } else {
      Write-Step "Updating existing SciAtlas checkout"
      git -C $TargetDir fetch --all --tags --prune
      if ($CheckoutRef) {
        git -C $TargetDir checkout $CheckoutRef
      } else {
        git -C $TargetDir pull --ff-only
      }
      return
    }
  }

  Write-Step "Downloading SciAtlas repository"
  git clone $RepositoryUrl $TargetDir
  if ($CheckoutRef) {
    git -C $TargetDir checkout $CheckoutRef
  }
}

$targetDir = Resolve-TargetPath $InstallDir

Write-Host "SciAtlas uv installer"
Write-Host "Repository : $RepoUrl"
Write-Host "Ref        : $(if ($Ref) { $Ref } else { "(default branch)" })"
Write-Host "Install dir: $targetDir"

Ensure-Uv
Ensure-Git
Sync-Repository -TargetDir $targetDir -RepositoryUrl $RepoUrl -CheckoutRef $Ref

Write-Step "Creating uv virtual environment"
Push-Location $targetDir
try {
  uv venv

  Write-Step "Installing SciAtlas CLI and workflow dependencies into .venv"
  uv pip install -e ".\sciatlas"
  uv pip install -r ".\requirements-workflows.txt"

  if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
  }

  if (-not $SkipToolInstall) {
    Write-Step "Installing editable global sciatlas command with workflow dependencies"
    uv tool install --editable --with-requirements ".\requirements-workflows.txt" --force ".\sciatlas"
  }

  $localCli = Join-Path $targetDir ".venv\Scripts\sciatlas.exe"
  if (Test-Path $localCli) {
    & $localCli -h | Out-Null
  }
} finally {
  Pop-Location
}

Write-Host ""
Write-Host "SciAtlas is ready."
Write-Host "Repository: $targetDir"
Write-Host "Local CLI : $targetDir\.venv\Scripts\sciatlas.exe"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cd `"$targetDir`""
Write-Host "  notepad .env   # set SCIATLAS_API_KEY"
Write-Host "  .\.venv\Scripts\sciatlas.exe -h"
Write-Host ""
Write-Host "If uv tool installed successfully, you can also run:"
Write-Host "  sciatlas -h"
