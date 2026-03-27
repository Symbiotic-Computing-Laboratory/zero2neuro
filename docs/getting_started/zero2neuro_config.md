---
title: Installing and Configuring Zero2Neuro
nav_order: 30
has_children: true
parent: Getting Started
---

# Installing Configuring Zero2Neuro

## 1. Create a directory (folder) that will hold copies of the needed
repositories

Bash shell:
```
mkdir z2n
```

## 2. Change your working directory:

Bash shell:

```
cd z2n
```

## 3. Clone the needed repositories

Clone both the Zero2Neuro and Keras3_Tools repositories.

- [Zero2Neuro](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro)
- [Keras 3 Tools](https://github.com/Symbiotic-Computing-Laboratory/keras3_tools)

Bash shell:

```
git clone git@github.com:Symbiotic-Computing-Laboratory/zero2neuro.git
git clone git@github.com:Symbiotic-Computing-Laboratory/keras3_tools.git
```

## 4. Set the NEURO_REPOSITORY_PATH environment variable

Bash shell:

```
export NEURO_REPOSITORY_PATH=/path/to/z2n
```

You can type this command by hand every time you use Zero2Neuro or you
can add the export command to your .bashrc file.

