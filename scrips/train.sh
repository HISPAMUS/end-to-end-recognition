#!/bin/sh

echo "Number of iter (list of iterations like '3 5 7')"
read -a iter

echo "Number of epochs"
read epochs

echo "Imput data lst"
read dataPath

echo "Test name"
read testName

echo "================================================"
echo ""
echo "Test: $testName"
echo ""

# Creacion de directorios para el test
echo "mkdir Model/$testName"
echo "mkdir res/$testName"

echo "python -u code/training.py --input-data $dataPath --output-vocabulary Model/$testName/SinModificaciones/vocabulary.npy --save-model Model/$testName/SinModificaciones/SinModificaciones --image-transformations 1 --conf-transformations-path modifierConf/SinModificacion --epochs $epochs > res/$testName/SinModificaciones.txt &"
wait

for i in ${iter[@]}
do
    echo "Solo contraste $i itaracion/es"
    echo "mkdir Model/$testName/SoloContraste"
    echo "python -u code/training.py --input-data $dataPath --output-vocabulary Model/$testName/SoloContraste/vocabulary.npy --save-model Model/$testName/SoloContraste/SoloContraste --image-transformations $i --conf-transformations-path modifierConf/SoloContraste --epochs $epochs > res/$testName/SoloContraste.txt"
done