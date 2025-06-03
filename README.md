# Motion Metrics

Identify human daily activities using on-device IMU and nerual network models, using [Motion Sense](https://github.com/mmalekzadeh/motion-sense.git) [dataset](https://www.kaggle.com/api/v1/datasets/download/malekzadeh/motionsense-dataset)

At the top level, you can get the full list of parameters by running:

```bash
cd path/to/this/repo && python src/python/training.py --help
```

For example, run:

```bash
cd path/to/this/repo && python src/python/training.py --epochs 32 --lstm-units 128 --dense-units=64 --batch-size 128
```

will train a model with 32 epochs, 128 LSTM units, 64 dense units, and a batch size of 128.