# Motion Metrics

Identify human daily activities using on-device IMU and nerual network models, using [Motion Sense](https://github.com/mmalekzadeh/motion-sense.git) [dataset](https://www.kaggle.com/api/v1/datasets/download/malekzadeh/motionsense-dataset)

At the top level, you can get the full list of parameters by running:

```bash
cd path/to/this/repo && python src/python/training.py --help
```

For example, run:

```bash
cd path/to/this/repo && python src/python/training.py --epochs 150 --batch-size 32
```

or directly:

```bash 
cd path/to/this/repo && python src/python/training.py 150 32
```

will train a model with 150 epochs and a batch size of 32.