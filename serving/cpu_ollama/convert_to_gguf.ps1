<#
 Convert the MERGED fp16 Hugging Face model folder -> GGUF (Windows PowerShell).

   .\convert_to_gguf.ps1 -ModelDir C:\models\finai-merged [-OutType q8_0|f16|bf16] [-OutDir .]

 q4_K_M needs llama-quantize.exe (download a Windows release zip from
 https://github.com/ggml-org/llama.cpp/releases), so on Windows the easy path is: convert to q8_0 here.
 Needs: git, Python 3 on PATH.
#>
param(
  [Parameter(Mandatory = $true)][string]$ModelDir,
  [ValidateSet("q8_0", "f16", "bf16")][string]$OutType = "q8_0",
  [string]$OutDir = $PSScriptRoot
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path "$ModelDir\config.json")) { throw "config.json not found in $ModelDir - pass the merged model folder." }
if (Test-Path "$ModelDir\adapter_config.json") { throw "$ModelDir looks like a LoRA adapter; merge it into the base model first." }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Llama = if ($env:LLAMA_CPP_DIR) { $env:LLAMA_CPP_DIR } else { Join-Path $PSScriptRoot "llama.cpp" }
if (-not (Test-Path $Llama)) {
  Write-Host ">> cloning llama.cpp into $Llama"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp $Llama
}
Write-Host ">> installing converter requirements"
python -m pip install -q -r "$Llama\requirements\requirements-convert_hf_to_gguf.txt"

$Out = Join-Path $OutDir "finai-$OutType.gguf"
python "$Llama\convert_hf_to_gguf.py" $ModelDir --outfile $Out --outtype $OutType
Write-Host ""
Write-Host "Done: $Out"
Write-Host "Next: put it next to the Modelfile (edit the FROM line if the name differs), then:  .\run_ollama.ps1"
