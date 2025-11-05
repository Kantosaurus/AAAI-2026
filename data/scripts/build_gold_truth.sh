#!/bin/bash
# Master script to build complete gold truth dataset
# This orchestrates fetching NVD data, generating synthetics, and combining them

set -e  # Exit on error

echo "================================================="
echo "Building Gold Truth CVE Dataset"
echo "================================================="
echo ""

# Configuration
DATA_DIR="../"
SCRIPTS_DIR="."
OUTPUT_DIR="../outputs"

CVE_LIST="${DATA_DIR}/cve_list_important.txt"
NVD_OUTPUT="${OUTPUT_DIR}/nvd_metadata.json"
SYNTHETIC_OUTPUT="${OUTPUT_DIR}/synthetic_cves.json"
GOLD_TRUTH_OUTPUT="${OUTPUT_DIR}/gold_truth_cves.json"
GOLD_TRUTH_CSV="${OUTPUT_DIR}/gold_truth_cves.csv"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Step 1: Fetch real CVE metadata from NVD
echo "Step 1: Fetching real CVE metadata from NVD..."
echo "  Source: NIST National Vulnerability Database"
echo "  CVE list: ${CVE_LIST}"
echo ""

if [ -f "${NVD_OUTPUT}" ]; then
    echo "  ⚠ NVD metadata already exists at ${NVD_OUTPUT}"
    read -p "  Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Using existing NVD metadata"
    else
        python3 "${SCRIPTS_DIR}/fetch_nvd_metadata.py" \
            --cve-list "${CVE_LIST}" \
            --output "${NVD_OUTPUT}"
    fi
else
    python3 "${SCRIPTS_DIR}/fetch_nvd_metadata.py" \
        --cve-list "${CVE_LIST}" \
        --output "${NVD_OUTPUT}"
fi

echo ""
echo "✓ Step 1 complete"
echo ""

# Step 2: Generate synthetic non-existent CVEs
echo "Step 2: Generating synthetic non-existent CVEs..."
echo "  Purpose: Hallucination testing"
echo "  Count: 100 synthetic CVEs"
echo ""

python3 "${SCRIPTS_DIR}/generate_synthetic_cves.py" \
    --output "${SYNTHETIC_OUTPUT}" \
    --count 100 \
    --seed 42

echo ""
echo "✓ Step 2 complete"
echo ""

# Step 3: Combine into gold truth dataset
echo "Step 3: Creating unified gold truth dataset..."
echo "  Combining real and synthetic CVEs"
echo ""

python3 "${SCRIPTS_DIR}/create_gold_truth_dataset.py" \
    --real "${NVD_OUTPUT}" \
    --synthetic "${SYNTHETIC_OUTPUT}" \
    --output "${GOLD_TRUTH_OUTPUT}" \
    --csv "${GOLD_TRUTH_CSV}" \
    --seed 42

echo ""
echo "✓ Step 3 complete"
echo ""

# Summary
echo "================================================="
echo "✓ Gold Truth Dataset Build Complete"
echo "================================================="
echo ""
echo "Output files:"
echo "  - Real CVEs (NVD):     ${NVD_OUTPUT}"
echo "  - Synthetic CVEs:      ${SYNTHETIC_OUTPUT}"
echo "  - Gold Truth (JSON):   ${GOLD_TRUTH_OUTPUT}"
echo "  - Gold Truth (CSV):    ${GOLD_TRUTH_CSV}"
echo ""
echo "Next steps:"
echo "  1. Review ${GOLD_TRUTH_OUTPUT} for accuracy"
echo "  2. Use with LLM testing framework"
echo "  3. Update quarterly with: ./update_nvd_metadata.sh"
echo ""
