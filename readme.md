
# nescient
## About
A convolutional neural network designed to detect lung lesions in chest radiographs. Read the [writeup](./ml-project-1.pdf) for a more in-depth explanation!

## Usage
Warning: this project relies on being on an Intel system with GPUs that have CUDA support!

### Dependencies
You will need to install Poetry if you do not already have it. Follow the instructions [here](https://github.com/python-poetry/poetry#installation) to get it.

From there, you can run `poetry install` to install the needed dependencies. Then, run `poetry shell` to enter a virtual environment.

### Model
You'll need a copy of the CheXpert dataset, which can be obtained [here](https://stanfordmlgroup.github.io/competitions/chexpert/). 

After that, you can run `preprocess.py` like so: 
```
python preprocess.py <path-to-train-csv>
```

This will generate a file called `all.csv` within the current directory. To begin training the model you can now run the following command:
```
python main.py ./all.csv <path-to-data-folder>
```
For an example: I installed CheXpert to `~/aux`, so that's what I would put as my data folder.

## Testing
You can run tests for `nescient` by running `tox -e ALL`. This will run `pytest` to run tests, `interrogate` to checks the documentation coverage, and `mypy` to do type checking.

If you're in the mood for some experimentation, `nescient` also installs `py-spy` as a dev dependency for profiling information.

## Documentation
You can generate documentation for `nescient` by running `pdoc nescient`.
