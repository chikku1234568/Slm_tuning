<#
 Create the `finai` model in Ollama from .\Modelfile, then smoke-test one narration through Ollama's
 OpenAI-compatible endpoint and print the latency (Windows PowerShell 5.1+ / PowerShell 7).

   .\run_ollama.ps1
   .\run_ollama.ps1 -Model finai -Url http://localhost:11434
   .\run_ollama.ps1 -SkipCreate -Model qwen2.5:0.5b      # test an already-pulled model
#>
param(
  [string]$Model = "finai",
  [string]$Url = "http://localhost:11434",
  [string]$Narration = "NACH-DR-BAJAJFIN-1234567890",
  [switch]$SkipCreate
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "ollama not found. Install: https://ollama.com/download" }
try { Invoke-WebRequest -UseBasicParsing "$Url/" | Out-Null } catch { throw "Ollama server not reachable at $Url. Start the Ollama app or run: ollama serve" }

if (-not $SkipCreate) {
  $fromFile = ((Select-String -Path "$PSScriptRoot\Modelfile" -Pattern '^FROM\s+(.+)$' | Select-Object -First 1).Matches.Groups[1].Value).Trim()
  if (-not (Test-Path (Join-Path $PSScriptRoot $fromFile))) { throw "GGUF '$fromFile' not found in $PSScriptRoot. Convert one first (README, step 2)." }
  Write-Host ">> ollama create $Model -f Modelfile"
  Push-Location $PSScriptRoot
  try { ollama create $Model -f Modelfile } finally { Pop-Location }
}

$sys = "Convert the bank statement narration into JSON with keys: category, sub_category, counterparty_name, counterparty_type, is_emi, lender_name, is_bounce_or_return, is_reversal, p2p_or_p2m, confidence. Output JSON only."
# Same JSON schema as serving/schema.py
$schema = '{"type":"object","additionalProperties":false,"required":["category","sub_category","counterparty_name","counterparty_type","is_emi","lender_name","is_bounce_or_return","is_reversal","p2p_or_p2m","confidence"],"properties":{"category":{"type":"string"},"sub_category":{"type":"string"},"counterparty_name":{"type":["string","null"]},"counterparty_type":{"enum":["merchant","individual","lender","employer","bank","unknown"]},"is_emi":{"type":"boolean"},"lender_name":{"type":["string","null"]},"is_bounce_or_return":{"type":"boolean"},"is_reversal":{"type":"boolean"},"p2p_or_p2m":{"enum":["P2P","P2M","NA"]},"confidence":{"type":"number","minimum":0,"maximum":1}}}' | ConvertFrom-Json
$body = @{
  model = $Model; temperature = 0; max_tokens = 200
  messages = @(@{ role = "system"; content = $sys }, @{ role = "user"; content = $Narration })
  response_format = @{ type = "json_schema"; json_schema = @{ name = "narration"; schema = $schema } }
} | ConvertTo-Json -Depth 20

function Call-Once {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-RestMethod -Method Post -Uri "$Url/v1/chat/completions" -ContentType "application/json" -Body $body
  $sw.Stop()
  [pscustomobject]@{ Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2); Text = $r.choices[0].message.content }
}
Write-Host ">> warm-up call (first call loads the model into RAM; NOT representative latency)"
$w = Call-Once; Write-Host "   first call: $($w.Seconds)s"
Write-Host ">> timed calls (narration: $Narration)"
$last = $null
1..3 | ForEach-Object { $last = Call-Once; Write-Host "   call ${_}: $($last.Seconds)s" }
Write-Host ">> response:"; Write-Host $last.Text
Write-Host ""
Write-Host "Latency above = wall time of one request on THIS machine (CPU, 1 request at a time). Measure with your own narrations."
Write-Host "Next: `$env:SLM_URL='$Url'; `$env:SLM_MODEL='$Model'; `$env:SLM_GUIDED='ollama'; streamlit run ..\streamlit_app.py"
