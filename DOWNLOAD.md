Dataset **BrackishMOT** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/e/o/Xn/wuNaOfe23iHQZYmVpMMU44443V2jUyWTgyT3sR04Di1GUpaUrGJW41tQ4hrxtVe48cD9EMy6nWS1TXYpszLpI8oWvXOyi1HysYmChX80Ew1beEGHSHwMw3VPWYWu.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='BrackishMOT', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://www.kaggle.com/datasets/maltepedersen/brackishmot/download?datasetVersionNumber=4).