VAE-GAN

Reconstructing the paper "Autoencoding beyond pixels using a leraned similarity metric"


To print an Image:
```
python utils.py 
```

For how to call the function see *run_interactive.ipynb* 

Train:
```
python train.py 
```

Saving models:
- The ID of a model is {time_of_running + model_name}. The same ID is used everywhere.

./saved_models - weights of the model are saved in a directory identified by ID, with the extension of the model
./logs/training.log - search the identifier to find its hyper-parameters as being specified in the runtime args
./tensorboard - same ID is used to look into the images
