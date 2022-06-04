FOLDER_PATH="data"
LOFTR_FOLDER_PATH="${FOLDER_PATH}/kornia-loftr"
SUPERGLUE_FOLDER_PATH="${FOLDER_PATH}/super-glue-pretrained-network"
mkdir -p ${FOLDER_PATH}
mkdir -p ${LOFTR_FOLDER_PATH}
mkdir -p ${SUPERGLUE_FOLDER_PATH}
cd ${FOLDER_PATH}
kaggle datasets download -d ammarali32/kornia-loftr
kaggle datasets download -d losveria/super-glue-pretrained-network
unzip kornia-loftr.zip LOFTR_FOLDER_PATH
unzip super-glue-pretrained-network.zip SUPERGLUE_FOLDER_PATH
cd ..
