# Why `custom_network.py` Exists

## The Short Answer

`plugin_manager.py` is a **generic loader** — it knows nothing about Keras or neural networks.  
`custom_network.py` is a **Keras-specific adapter** — it enforces the contract that the network pipeline needs.

They solve two completely different problems.

---

## What `plugin_manager.py` Does

It is pure infrastructure. Given any plugin name, it:

1. Finds the `.py` file on disk
2. Loads it with `importlib`
3. Instantiates the class inside it
4. Calls `plugin.call(**kwargs)`

It is fully **domain-blind**. It has zero knowledge of Keras, `build_model`, model compilation, or what a neural network even is. It works the same way for a data-preprocessing plugin, a model-wrapper plugin, or any future plugin.

---

## The Gap It Cannot Fill

When `--network_type plugin` is set, the training pipeline needs a **compiled `keras.Model`** to come out the other side.

`plugin_manager` alone cannot handle this because it does not know:

- That the user must provide a `network_file=` path pointing to a second `.py` file
- That this second `.py` must contain a function named exactly `build_model(args)`
- That `args` (the full Zero2Neuro config namespace) must be passed into it
- That the return value must be a valid, **compiled** `keras.Model`

---

## What `custom_network.py` Adds

It acts as the bridge between the generic loader and the user's raw architecture file.

| Responsibility | `plugin_manager.py` | `custom_network.py` |
|---|:---:|:---:|
| Find and load a plugin `.py` by class name | ✅ | — |
| Validate the `network_file` path exists and is `.py` | — | ✅ |
| Dynamically load the **user's architecture `.py`** | — | ✅ |
| Enforce that `build_model(args)` exists | — | ✅ |
| Pass the full `args` namespace into `build_model` | — | ✅ |
| Validate the return is a compiled `keras.Model` | — | ✅ |
| Wrap the result as `{'model': result}` for the pipeline | — | ✅ |

---

## The Two-Level Loading Chain

```
plugin_manager.py
    └── loads → custom_network.py
                    └── loads → user's my_network.py
                                    └── returns compiled keras.Model
```

`plugin_manager` handles **level 1** (loading the plugin).  
`custom_network` handles **level 2** (loading the user's architecture and enforcing the Keras contract).

Without `custom_network.py`, the pipeline would have no way to know what a valid result looks like, or how to find and run the user's architecture file.
