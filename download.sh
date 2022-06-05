FOLDER_PATH="data"
mkdir -p ${FOLDER_PATH}
cd ${FOLDER_PATH}
kaggle competitions download -c image-matching-challenge-2022
kaggle datasets download -d ammarali32/kornia-loftr
kaggle datasets download -d losveria/super-glue-pretrained-network
unzip image-matching-challenge-2022.zip -d image-matching-challenge-2022
unzip kornia-loftr.zip -d kornia-loftr
unzip super-glue-pretrained-network.zip -d models/SuperGluePretrainedNetwork
cp super-glue-pretrained-network/models/superglue.py LoFTR/src/loftr/utils/
rm image-matching-challenge-2022.zip
rm super-glue-pretrained-network.zip
rm kornia-loftr.zip
cd ..