python eval.py --test True 
python utils/init_vertex_extraction.py
mkdir -p ./dataset/init_vertices
cp -r ./records/endpoint/vertices/* ./dataset/init_vertices/