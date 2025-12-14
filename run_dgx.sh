#!/bin/bash
# ============================================================================
# DGX Spark Frontier Hackathon - GPU Training Runner
# Port-to-Rail Surge Forecaster
# ============================================================================
#
# This script runs the complete GPU-accelerated training pipeline on DGX.
# It demonstrates NVIDIA GPU utilization across multiple frameworks.
#
# Usage:
#   ./run_dgx.sh              # Run full training pipeline
#   ./run_dgx.sh --benchmark  # Run CPU vs GPU benchmark
#   ./run_dgx.sh --verify     # Verify GPU setup only
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║   🚢 PORT-TO-RAIL SURGE FORECASTER                                          ║"
echo "║   DGX Spark Frontier Hackathon 2025 - Glīd Partner Challenge                ║"
echo "║                                                                              ║"
echo "║   GPU-Accelerated Machine Learning Pipeline                                  ║"
echo "║   • RAPIDS cuDF/cuML - 100x faster data processing                          ║"
echo "║   • XGBoost gpu_hist - GPU gradient boosting                                ║"
echo "║   • PyTorch CUDA - Deep learning LSTM                                        ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# GPU Detection
# ============================================================================
echo -e "${BLUE}[1/5] Detecting NVIDIA GPUs...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ nvidia-smi found${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found - GPU may not be available${NC}"
    GPU_COUNT=0
fi

# ============================================================================
# Python Environment Check
# ============================================================================
echo -e "\n${BLUE}[2/5] Checking Python environment...${NC}"

check_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓ $1: $VERSION${NC}"
        return 0
    else
        echo -e "${YELLOW}○ $1: not installed${NC}"
        return 1
    fi
}

echo "Checking GPU packages:"
check_package torch && python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
check_package cudf || true
check_package cuml || true
check_package xgboost

# ============================================================================
# CUDA Verification
# ============================================================================
echo -e "\n${BLUE}[3/5] Verifying CUDA support...${NC}"

python << 'EOF'
import sys

# PyTorch CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ PyTorch CUDA: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("○ PyTorch CUDA: Not available")
except ImportError:
    print("○ PyTorch: Not installed")

# RAPIDS
try:
    import cudf
    import cuml
    print(f"✓ RAPIDS cuDF: {cudf.__version__}")
    print(f"✓ RAPIDS cuML: {cuml.__version__}")
except ImportError:
    print("○ RAPIDS: Not installed")

# XGBoost GPU
try:
    import xgboost as xgb
    import numpy as np
    # Test GPU training
    dtrain = xgb.DMatrix(np.random.rand(100, 10), label=np.random.rand(100))
    params = {'tree_method': 'gpu_hist', 'device': 'cuda'}
    try:
        xgb.train(params, dtrain, num_boost_round=1)
        print("✓ XGBoost GPU: Available (gpu_hist)")
    except:
        print("○ XGBoost GPU: Not available")
except ImportError:
    print("○ XGBoost: Not installed")
EOF

# ============================================================================
# Run Mode Selection
# ============================================================================
echo -e "\n${BLUE}[4/5] Selecting run mode...${NC}"

MODE="train"
if [[ "$1" == "--benchmark" ]]; then
    MODE="benchmark"
    echo -e "${CYAN}Mode: CPU vs GPU Benchmark${NC}"
elif [[ "$1" == "--verify" ]]; then
    MODE="verify"
    echo -e "${CYAN}Mode: Verify GPU Setup Only${NC}"
else
    echo -e "${CYAN}Mode: Full GPU Training Pipeline${NC}"
fi

# ============================================================================
# Run Training
# ============================================================================
echo -e "\n${BLUE}[5/5] Running pipeline...${NC}"
echo "============================================================================"

if [[ "$MODE" == "verify" ]]; then
    echo -e "${GREEN}GPU verification complete!${NC}"
    exit 0
fi

if [[ "$MODE" == "benchmark" ]]; then
    echo -e "${CYAN}Starting CPU vs GPU benchmark...${NC}"
    python gpu_training_showcase.py --benchmark
else
    echo -e "${CYAN}Starting enhanced GPU training...${NC}"
    python train_enhanced_model.py --gpu
fi

echo ""
echo "============================================================================"
echo -e "${GREEN}Pipeline complete!${NC}"
echo "============================================================================"

# ============================================================================
# Results Summary
# ============================================================================
echo -e "\n${CYAN}Results saved to:${NC}"
echo "  • models/champion_*.json    - Trained models"
echo "  • output/enhanced_*.json    - Training report"
echo "  • output/gpu_benchmark*.json - Benchmark results (if run)"

echo -e "\n${CYAN}To start the API:${NC}"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo -e "\n${CYAN}To start the dashboard:${NC}"
echo "  cd dashboard && npm run dev"
