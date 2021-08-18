mkdir -p ./cropped_tiff
mkdir -p ./temp_raw_tiff

wget -O Queens.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_queens_sp16.zip
unzip -d ./Queens Queens.zip
rm -rf ./Queens.zip
mv ./Queens/*.jp2 ./temp_raw_tiff
rm -rf ./Queens

wget -O Manhattan.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_manhattan_sp16.zip
unzip -d ./Manhattan Manhattan.zip
rm -rf ./Manhattan.zip
mv ./Manhattan/*.jp2 ./temp_raw_tiff
rm -rf ./Manhattan

wget -O Kings.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_brooklyn_sp16.zip
unzip -d ./Kings Kings.zip
rm -rf ./Kings.zip
mv ./Kings/*.jp2 ./temp_raw_tiff
rm -rf ./Kings

wget -O Richmond.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_staten_island_sp16.zip
unzip -d ./Richmond Richmond.zip
rm -rf ./Richmond.zip
mv ./Richmond/*.jp2 ./temp_raw_tiff
rm -rf ./Richmond

wget -O Bronx.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_bronx_sp16.zip
unzip -d ./Bronx Bronx.zip
rm -rf ./Bronx.zip
mv ./Bronx/*.jp2 ./temp_raw_tiff
rm -rf ./Bronx

python3 ./scripts/generate_cropped_tiff.py

rm -rf ./temp_raw_tiff