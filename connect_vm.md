# Conenct to VM

## 1. Connect to UDST Global Protect VPN 

Check Parallel Lab 1 Student Guide to Global Protect VPN to set it up

## 2. SSH into the VM

Open PowerShell or your terminal and run:

```bash

ssh student@10.125.85.10

```

## 3. Start Jupyter Notebook on the VM

Inside the VM, run:

```bash

source notebook-env/bin/activate

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888

```

## 4. Create an SSH Tunnel from Your Laptop

Open **a second** terminal on your laptop and run:

```bash

ssh -L8888:localhost:8888 student@10.125.85.10

```

## (Optional) Check GPU config

```bash

! nvidia-smi

```
