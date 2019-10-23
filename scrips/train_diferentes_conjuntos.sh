for pages in 10 20 30 40 50 60 70 80 90 100
do
    echo "==================================="
    echo "======== Prueba $pages paginas ========"
    echo "==================================="
    echo ""

    mkdir Model/B-59.850_$pages
    mkdir res/B-59.850_$pages

    echo "########## Sin modificaciones ##########"
    echo ""

    mkdir Model/B-59.850_$pages/SinModificaciones
    python -u code/training.py --input-data data/B-59.850_$pages.lst --output-vocabulary Model/B-59.850_$pages/SinModificaciones/vocabulary.npy --save-model Model/B-59.850_$pages/SinModificaciones/SinModificaciones --image-transformations 1 --conf-transformations-path modifierConf/SinModificacion --epochs 300 > res/B-59.850_$pages/SinModificaciones.txt

    echo "########## Con modificaciones ##########"
    echo ""

    mkdir Model/B-59.850_$pages/ConModificaciones
    python -u code/training.py --input-data data/B-59.850_$pages.lst --output-vocabulary Model/B-59.850_$pages/ConModificaciones/vocabulary.npy --save-model Model/B-59.850_$pages/ConModificaciones/ConModificaciones --image-transformations 3 --conf-transformations-path modifierConf/all --epochs 300 > res/B-59.850_$pages/ConModificaciones.txt
done
