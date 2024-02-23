torchserve --stop
rm -r logs
rm model_store/voice2face.mar
torch-model-archiver --model-name voice2face --version 0.1 --handler handler.py 
mv voice2face.mar ./model_store
torchserve --start --ncs --model-store model_store --models voice2face.mar --ts-config config.properties