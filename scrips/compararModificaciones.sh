#!/bin/sh

mkdir Model/B-50.747
mkdir res/B-50.747

echo "==========================================="
echo "======== Prueba sin modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/B-50.747/SinModificaciones
python -u code/training.py --input-data data/B-50.747.lst --output-vocabulary Model/B-50.747/SinModificaciones/vocabulary.npy --save-model Model/B-50.747/SinModificaciones/SinModificaciones --image-transformations 1 --conf-transformations-path modifierConf/SinModificacion --epochs 300 > res/B-50.747/SinModificaciones.txt

echo "==========================================="
echo "======== Prueba con modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/B-50.747/ConModificaciones
python -u code/training.py --input-data data/B-50.747.lst --output-vocabulary Model/B-50.747/ConModificaciones/vocabulary.npy --save-model Model/B-50.747/ConModificaciones/ConModificaciones --image-transformations 3 --conf-transformations-path modifierConf/all --epochs 300 > res/B-50.747/ConModificaciones.txt

mkdir Model/B-53.781
mkdir res/B-53.781

echo "==========================================="
echo "======== Prueba sin modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/B-53.781/SinModificaciones
python -u code/training.py --input-data data/B-53.781.lst --output-vocabulary Model/B-53.781/SinModificaciones/vocabulary.npy --save-model Model/B-53.781/SinModificaciones/SinModificaciones --image-transformations 1 --conf-transformations-path modifierConf/SinModificacion --epochs 300 > res/B-53.781/SinModificaciones.txt

echo "==========================================="
echo "======== Prueba con modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/B-53.781/ConModificaciones
python -u code/training.py --input-data data/B-53.781.lst --output-vocabulary Model/B-53.781/ConModificaciones/vocabulary.npy --save-model Model/B-53.781/ConModificaciones/ConModificaciones --image-transformations 3 --conf-transformations-path modifierConf/all --epochs 300 > res/B-53.781/ConModificaciones.txt

mkdir Model/B-3.28
mkdir res/B-3.28

echo "==========================================="
echo "======== Prueba sin modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/B-3.28/SinModificaciones
python -u code/training.py --input-data data/B-3.28.lst --output-vocabulary Model/B-3.28/SinModificaciones/vocabulary.npy --save-model Model/B-3.28/SinModificaciones/SinModificaciones --image-transformations 1 --conf-transformations-path modifierConf/SinModificacion --epochs 300 > res/B-3.28/SinModificaciones.txt

echo "==========================================="
echo "======== Prueba con modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/B-3.28/ConModificaciones
python -u code/training.py --input-data data/B-3.28.lst --output-vocabulary Model/B-3.28/ConModificaciones/vocabulary.npy --save-model Model/B-3.28/ConModificaciones/ConModificaciones --image-transformations 3 --conf-transformations-path modifierConf/all --epochs 300 > res/B-3.28/ConModificaciones.txt