#!/bin/bash
set -euo pipefail

# 🔥 Ultra-CNN Complete Workflow Script
# 🎯 Train → Evaluate → API workflow for Ultra-CNN
# 📈 95% accuracy target pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "🔥 Ultra-CNN Complete Workflow"
echo "🎯 Target: 95% accuracy (challenging Transformer)"
echo "🏗️ Architecture: ResNet + CBAM + FPN"
echo "=============================================="
echo ""

# Parse command line arguments
TRAIN_ONLY=false
EVALUATE_ONLY=false
API_ONLY=false
SKIP_TRAIN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --train-only)
      TRAIN_ONLY=true
      shift
      ;;
    --evaluate-only)
      EVALUATE_ONLY=true
      shift
      ;;
    --api-only)
      API_ONLY=true
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--train-only|--evaluate-only|--api-only|--skip-train]"
      exit 1
      ;;
  esac
done

# Step 1: Training
if [ "$EVALUATE_ONLY" = false ] && [ "$API_ONLY" = false ]; then
  if [ "$SKIP_TRAIN" = false ]; then
    echo "🚀 Step 1: Ultra-CNN Training"
    echo "⏱️  This may take 30-60 minutes depending on hardware..."
    echo ""

    if ! ./scripts/train_ultra_cnn.sh; then
      echo "❌ Training failed! Check logs above."
      exit 1
    fi

    echo ""
    echo "✅ Ultra-CNN training completed successfully!"
    echo ""
  else
    echo "⏭️  Skipping training (--skip-train specified)"
    echo ""
  fi
fi

# Step 2: Evaluation
if [ "$TRAIN_ONLY" = false ] && [ "$API_ONLY" = false ]; then
  echo "📊 Step 2: Ultra-CNN Evaluation"
  echo ""

  if ! ./scripts/evaluate_ultra_cnn.sh; then
    echo "❌ Evaluation failed! Check if model exists."
    exit 1
  fi

  echo ""
  echo "✅ Ultra-CNN evaluation completed!"
  echo ""
fi

# Step 3: API Service (optional)
if [ "$TRAIN_ONLY" = false ] && [ "$EVALUATE_ONLY" = false ]; then
  echo "🌐 Step 3: Ultra-CNN API Service"
  echo "⚠️  This will start the API server (Ctrl+C to stop)"
  echo ""

  read -p "🤔 Start Ultra-CNN API server? (y/N): " -n 1 -r
  echo ""

  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Ultra-CNN API server..."
    echo ""
    ./scripts/api_ultra_cnn.sh
  else
    echo "⏭️  Skipping API server startup."
  fi
fi

echo ""
echo "🎉 Ultra-CNN Workflow Completed!"
echo "==============================================="
echo "📈 Performance Comparison:"
echo "  - Basic CNN:     0%      (failed)"
echo "  - Optimized CNN: 71.26%  (breakthrough)"
echo "  - Ultra-CNN:     [Check evaluation results]"
echo "  - Transformer:   93.98%  (target to beat)"
echo ""
echo "🔧 Available scripts:"
echo "  - ./scripts/train_ultra_cnn.sh     (Training only)"
echo "  - ./scripts/evaluate_ultra_cnn.sh  (Evaluation only)"
echo "  - ./scripts/api_ultra_cnn.sh       (API service only)"
echo "  - ./scripts/ultra_cnn_workflow.sh  (Complete workflow)"