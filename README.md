# gnn-semi-supervised-text-class


## Setup
Create environment and install dependencies
```
conda create -n my-env python=3.8
```


Install PyTorch Geometric
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.1+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.1+cu102.html
pip install torch-geometric
```
Install other requirements
```
pip install -r requirements.txt
```


## Run
```
python main.py --my-arg my-value
```

See all the `run_my_exp.sh` files for examples of how to run the code.