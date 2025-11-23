# Conenct to VM

## 1. SSH into the VM

Open PowerShell or your terminal and run:

```bash

ssh student@10.125.85.10

```

## 2. Start Jupyter Notebook on the VM

Inside the VM, run:

```bash

source notebook-env/bin/activate

jupyter notebook--no-browser--ip=0.0.0.0--port=8888

```

## 3. Create an SSH Tunnel from Your Laptop

Open **a second** terminal on your laptop and run:

```bash

ssh -L8888:localhost:8888student@10.125.85.10

```

## 4 Check GPU config
```bash

! nvidia-smi

```
