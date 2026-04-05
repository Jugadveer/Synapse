$ErrorActionPreference = 'Stop'

oumi merge-lora `
  --base Qwen/Qwen2.5-3B-Instruct `
  --lora .\artifacts\qwen25-3b-intent-lora `
  --output .\artifacts\qwen25-3b-intent-merged

Write-Host "Merged model written to .\artifacts\qwen25-3b-intent-merged"
