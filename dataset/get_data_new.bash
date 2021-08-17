mkdir -p ./cropped_tiff
mkdir -p ./temp_raw_tiff

wget -O 2018_Manhattan_Orthoimagery_jp2.zip https://data.cityofnewyork.us/download/hxws-3mbm/application%2Fzip
unzip -d ./2018_Manhattan_Orthoimagery_jp2 2018_Manhattan_Orthoimagery_jp2.zip
rm -rf ./2018_Manhattan_Orthoimagery_jp2.zip
mv ./2018_Manhattan_Orthoimagery_jp2/* ./temp_raw_tiff

wget -O 2018_Kings_Orthoimagery_jp2.zip https://data.cityofnewyork.us/download/q937-un2p/application%2Fzip
unzip -d ./2018_Kings_Orthoimagery_jp2 2018_Kings_Orthoimagery_jp2.zip
rm -rf ./2018_Kings_Orthoimagery_jp2.zip
mv ./2018_Kings_Orthoimagery_jp2/* ./temp_raw_tiff

wget -O 2018_Queens_Orthoimagery_jp2.zip https://data.cityofnewyork.us/download/dint-nfj5/application%2Fzip
unzip -d ./2018_Queens_Orthoimagery_jp2 2018_Queens_Orthoimagery_jp2.zip
rm -rf ./2018_Queens_Orthoimagery_jp2.zip
mv ./2018_Queens_Orthoimagery_jp2/* ./temp_raw_tiff

wget -O 2018_StatenIsland_Orthoimagery_jp2.zip https://data.cityofnewyork.us/download/ayph-gj7e/application%2Fzip
unzip -d ./2018_StatenIsland_Orthoimagery_jp2 2018_StatenIsland_Orthoimagery_jp2.zip
rm -rf ./2018_StatenIsland_Orthoimagery_jp2.zip
mv ./2018_StatenIsland_Orthoimagery_jp2/* ./temp_raw_tiff

wget -O 2018_Bronx_Orthoimagery_jp2.zip https://data.cityofnewyork.us/download/hxws-3mbm/application%2Fzip
unzip -d ./2018_Bronx_Orthoimagery_jp2 2018_Bronx_Orthoimagery_jp2.zip
rm -rf ./2018_Bronx_Orthoimagery_jp2.zip
mv ./2018_Bronx_Orthoimagery_jp2/* ./temp_raw_tiff

python3 ./scripts/generate_cropped_tiff.py

rm -rf ./temp_raw_tiff