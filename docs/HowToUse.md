# How to use

First, make sure you are in the project environment, e.g.:
```bash
conda activate dl_project
```

## Scripts
You can launch the scripts in `scripts/` with:
```bash
python scripts/run_script.py [--option VALUE]
```

See [Scripts](Scripts.md) for information about the available scripts.

Use the `--help` argument if you want information on the options.

## Notebooks
Start a jupyter notebook server:
```bash
cd notebooks
jupyter notebook
```

See [Notebooks](Notebooks.md) for information about the available notebooks.

### Notebooks through `ssh`
You need to launch jupyter on the server. You may want to use the `--no-browser` option and `--port 8889` the set a different port.
```bash
jupyter notebook --no-browser --port 8889
``` 

Then, on your local machine, connect the ports with:
```bash
ssh -NfL localhost:8889:localhost:8889 <user>@<address>
```
Adapt the ports, user and address to your needs.