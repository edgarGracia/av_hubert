git submodule init
git submodule update
cd fairseq
python -m pip install -e ./
cd ..
bash download_data.sh