#!/bin/bash
# Cleanup script for thesis project
# Removes temporary files and cache while preserving important data

set -e

PROJECT_DIR="/user/amarino/tesi_project_amarino"
cd "$PROJECT_DIR"

echo "=========================================="
echo "PULIZIA PROGETTO TESI"
echo "=========================================="
echo ""

# Counter
removed_count=0

# 1. Remove Python cache files
echo "🧹 Rimozione cache Python..."
find . -type d -name "__pycache__" -print0 | while IFS= read -r -d '' dir; do
    echo "   Rimuovo: $dir"
    rm -rf "$dir"
    ((removed_count++))
done

find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -print0 | while IFS= read -r -d '' file; do
    echo "   Rimuovo: $file"
    rm -f "$file"
    ((removed_count++))
done

echo "✅ Cache Python rimossa"
echo ""

# 2. Remove temporary files
echo "🧹 Rimozione file temporanei..."
find . -maxdepth 3 -type f \( -name "*.tmp" -o -name "*.bak" -o -name "*~" \) -print0 | while IFS= read -r -d '' file; do
    echo "   Rimuovo: $file"
    rm -f "$file"
    ((removed_count++))
done

echo "✅ File temporanei rimossi"
echo ""

# 3. Remove old nohup files (keep recent ones in logs/)
echo "🧹 Rimozione vecchi nohup files..."
find . -maxdepth 2 -name "nohup.out" -type f -print0 | while IFS= read -r -d '' file; do
    echo "   Rimuovo: $file"
    rm -f "$file"
    ((removed_count++))
done

echo "✅ Nohup files rimossi"
echo ""

# 4. Remove empty directories in results (except meeting folders)
echo "🧹 Rimozione cartelle vuote in results..."
find results/ -mindepth 1 -maxdepth 2 -type d -empty ! -path "*/MEETING_*" -print0 | while IFS= read -r -d '' dir; do
    echo "   Rimuovo cartella vuota: $dir"
    rmdir "$dir" 2>/dev/null || true
    ((removed_count++))
done

echo "✅ Cartelle vuote rimosse"
echo ""

# 5. Clean up old checkpoint duplicates (keep only best and latest)
echo "🧹 Pulizia checkpoint duplicati..."
# This is safer - we just report, don't delete automatically
find external/YOLOX/YOLOX_outputs -name "epoch_*.pth" -type f | head -20 | while read file; do
    echo "   📦 Checkpoint trovato: $file (mantieni solo best e latest)"
done

echo "✅ Checkpoint verificati (non rimossi automaticamente)"
echo ""

echo "=========================================="
echo "✅ PULIZIA COMPLETATA"
echo "=========================================="
echo ""
echo "📊 Struttura mantenuta:"
echo "   ✓ logs/ - tutti i log di training/tracking"
echo "   ✓ results/MEETING_*/ - risultati organizzati per meeting"
echo "   ✓ weights/ - checkpoint modelli"
echo "   ✓ yolox_finetuning/ - training YOLOX con plot"
echo "   ✓ docs/ - documentazione tesi"
echo ""
echo "🗑️  Rimossi: cache Python, file temporanei, nohup files"
