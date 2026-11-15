param(
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",   # must match torch.version.cuda
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    [string]$Arch = "8.9"   # RTX 40xx (Ada); 8.6 for RTX 30xx, 12.0 for RTX 50xx
)

# Builds MVDeTr's MultiScaleDeformableAttention CUDA op into demo_multiview\native (MVDeTr sources stay untouched).
# inference.py loads it from there; without it MVDeTr falls back to the much slower PyTorch implementation.
$ops = Join-Path $PSScriptRoot "..\MVDeTr\multiview_detector\models\ops"
$native = Join-Path $PSScriptRoot "native"
$temp = Join-Path $env:TEMP "thesis_demo_deform_attn_build"
if (-not (Test-Path $VcVars)) { throw "MSVC not found: $VcVars" }
if (-not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) { throw "CUDA toolkit not found: $CudaHome" }
New-Item -ItemType Directory -Force $native | Out-Null

$build = "set CUDA_HOME=$CudaHome&& set TORCH_CUDA_ARCH_LIST=$Arch&& set DISTUTILS_USE_SDK=1&& " +
         "cd /d `"$ops`" && python setup.py build_ext --build-lib `"$native`" --build-temp `"$temp`""
cmd /c "`"$VcVars`" >nul && $build"
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }
Get-ChildItem $native -Filter "MultiScaleDeformableAttention*.pyd" | ForEach-Object { "Built $($_.FullName)" }
