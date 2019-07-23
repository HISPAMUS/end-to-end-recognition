#!/bin/sh

echo "==========================================="
echo "======== Prueba sin modificaciones ========"
echo "==========================================="
echo ""

mkdir Model/SinModificaciones
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SinModificaciones/vocabulary.npy --save-model Model/SinModificaciones/SinModificaciones --image-transformations 1 --conf-transformations-path modifierConf/SinModificacion --epochs 200 > res/SinModificaciones.txt

echo "==========================================="
echo "======== Prueba con solo contraste ========"
echo "==========================================="
echo ""

mkdir Model/SoloContraste

echo "########  3 Iteraciones ########"
echo ""

mkdir Model/SoloContraste/NI3
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloContraste/NI3/vocabulary.npy --save-model Model/SoloContraste/NI3/SoloContraste --image-transformations 3 --conf-transformations-path modifierConf/Contrast --epochs 200 > res/SoloContrasteNI3.txt

echo "########  5 Iteraciones ########"
echo ""

mkdir Model/SoloContraste/NI5
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloContraste/NI5/vocabulary.npy --save-model Model/SoloContraste/NI5/SoloContraste --image-transformations 5 --conf-transformations-path modifierConf/Contrast --epochs 200 > res/SoloContrasteNI5.txt

echo "==========================================="
echo "======== Prueba con solo ero_dilat ========"
echo "==========================================="
echo ""

mkdir Model/SoloEro_Dilat

echo "########  3 Iteraciones ########"
echo ""

mkdir Model/SoloEro_Dilat/NI3
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloEro_Dilat/NI3/vocabulary.npy --save-model Model/SoloEro_Dilat/NI3/SoloEro_Dilat --image-transformations 3 --conf-transformations-path modifierConf/Ero_Dila --epochs 200 > res/SoloEro_DilatNI3.txt

echo "########  5 Iteraciones ########"
echo ""

mkdir Model/SoloEro_Dilat/NI5
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloEro_Dilat/NI5/vocabulary.npy --save-model Model/SoloEro_Dilat/NI5/SoloEro_Dilat --image-transformations 5 --conf-transformations-path modifierConf/Ero_Dila --epochs 200 > res/SoloEro_DilatNI5.txt

echo "==========================================="
echo "======== Prueba con solo rand_marg ========"
echo "==========================================="
echo ""

mkdir Model/SoloRandomMargin

echo "########  3 Iteraciones ########"
echo ""

mkdir Model/SoloRandomMargin/NI3
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloRandomMargin/NI3/vocabulary.npy --save-model Model/SoloRandomMargin/NI3/SoloRandomMargin --image-transformations 3 --conf-transformations-path modifierConf/Margin --epochs 200 > res/SoloRandomMarginNI3.txt

echo "########  5 Iteraciones ########"
echo ""

mkdir Model/SoloRandomMargin/NI5
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloRandomMargin/NI5/vocabulary.npy --save-model Model/SoloRandomMargin/NI5/SoloRandomMargin --image-transformations 5 --conf-transformations-path modifierConf/Margin --epochs 200 > res/SoloRandomMarginNI5.txt

echo "==========================================="
echo "======== Prueba con solo rnd_rotat ========"
echo "==========================================="
echo ""

mkdir Model/SoloRandomRotation

echo "########  3 Iteraciones ########"
echo ""

mkdir Model/SoloRandomRotation/NI3
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloRandomRotation/NI3/vocabulary.npy --save-model Model/SoloRandomRotation/NI3/SoloRandomRotation --image-transformations 3 --conf-transformations-path modifierConf/Rotation --epochs 200 > res/SoloRandomRotationNI3.txt

echo "########  5 Iteraciones ########"
echo ""

mkdir Model/SoloRandomRotation/NI5
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloRandomRotation/NI5/vocabulary.npy --save-model Model/SoloRandomRotation/NI5/SoloRandomRotation --image-transformations 5 --conf-transformations-path modifierConf/Rotation --epochs 200 > res/SoloRandomRotationNI5.txt

echo "============================================"
echo "======== Prueba con solo ojo de pez ========"
echo "============================================"
echo ""

mkdir Model/SoloFishEye

echo "########  3 Iteraciones ########"
echo ""

mkdir Model/SoloFishEye/NI3
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloFishEye/NI3/vocabulary.npy --save-model Model/SoloFishEye/NI3/SoloFishEye --image-transformations 3 --conf-transformations-path modifierConf/Fish_eye --epochs 200 > res/SoloFishEyeNI3.txt

echo "########  5 Iteraciones ########"
echo ""

mkdir Model/SoloFishEye/NI5
python -u code/training.py --input-data data/B-59.850.lst --output-vocabulary Model/SoloFishEye/NI5/vocabulary.npy --save-model Model/SoloFishEye/NI5/SoloFishEye --image-transformations 5 --conf-transformations-path modifierConf/Fish_eye --epochs 200 > res/SoloFishEyeNI5.txt