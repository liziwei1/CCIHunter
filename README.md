# UniIntent

A comment-code consistency detector in DApp ecosystems.

# Quickly start

Please ensure that your Python version is higher than 3.9.
And then run the command below in the console.

```shell
pip install -r requirements.txt
npm install
```

If you are using **Linux** or **MacOS**, run the extra command below in the console:

```shell
chmod 755 misc/solc/solc-linux-amd64-v0.8.21+commit.d9974bed
chmod 755 solc-macosx-amd64-v0.8.21+commit.d9974bed
```

After that, you can run a training process for our models:
```shell
python train.py \
  --data_path=/path/to/your/dataset \
  --report_step=5 \
  --gpu=True \
  --batch_size=32 \
  --epoch=1 \
  --hidden_channels=384 \
  --num_heads=6 \
  --num_layers=6 \
  --num_workers=16
```

And then, using the following command to perform downstream tasks:
```shell
python test.py \
  --pretrain_path=/path/to/your/pretrain/model
  --rf_path=/path/to/your/random_forest/model
  --data_path=/path/to/your/data/json
```
Note that when `test_type=0`, the script performs CCI detection,
while when `test_type=1` and `hash2` are available, the script performs code similarity evaluation.
