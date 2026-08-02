# Plugin Architecture: Why Plugins Live in Separate Files

## Question

> *Can we put all of the plugins directly inside `plugin_manager.py` rather than in separate plugin modules?*

## Answer: No — and the separate-file design is correct

---

## Role Separation

`plugin_manager.py` and the individual plugin files have fundamentally different responsibilities:

| | `plugin_manager.py` | `plugins/normalization.py`, etc. |
|---|---|---|
| **Role** | Lifecycle management: discover, load, wire, apply | Domain logic: do the actual work |
| **Knows about** | File paths, roles, argument parsing, calling order | Keras, sklearn, netCDF4, etc. |
| **Changes when** | The plugin protocol changes | The algorithm changes |

Merging them violates the **Single Responsibility Principle** — the manager would become responsible for both *orchestrating* plugins and *implementing* them.

---

## Five Reasons to Keep Plugins in Separate Files

### 1. Users Can Add Their Own Plugins Without Touching Core Code

The `--plugin_path` argument lets any user point to their own directory:

```
--plugin_path /home/user/my_plugins
--plugin_list my_loader-preprocess
```

If all plugins lived inside `plugin_manager.py`, a user would be forced to edit a core framework file to add theirs. Separate files preserve the **Open/Closed Principle**: the system is open for extension but closed for modification.

---

### 2. Heavy Dependencies Stay Isolated

Each plugin imports only what it needs:

- `normalization.py` → imports Keras
- `netcdf_loader.py` → imports netCDF4
- `custom_network.py` → imports whatever the user needs

If all of these lived in `plugin_manager.py`, every import would be required at startup — even when the user doesn't use those plugins. With separate files, the manager only imports a plugin **when it is actually requested**.

---

### 3. Each Plugin Can Be Tested Independently

```python
# Test normalization in isolation — no manager needed
from plugins.normalization import normalization

p = normalization()
result = p.call(ins_train=X_train, ins_val=X_val)
assert 'normalization_layer' in result
```

With a monolithic file, you cannot import or test one plugin without loading all the others.

---

### 4. The Manager Stays Maintainable as Plugins Grow

The system already has `normalization`, `netcdf_loader`, and `custom_network` — and is designed for users to add more. A single file containing 10+ plugin classes, each with its own argument parsers and domain logic, would grow to thousands of lines. Separate files make it straightforward to find, edit, review, and version each plugin independently.

---

### 5. This Is the Established Pattern for Plugin Systems

Every well-known plugin system keeps the *manager* separate from the *implementations*:

| System | Manager | Plugin |
|--------|---------|--------|
| pytest | `pluginmanager.py` | `conftest.py` / separate packages |
| Django | `AppConfig` registry | Individual `apps/` |
| VS Code | Extension host | Individual `.vsix` extension files |
| zero2neuro | `plugin_manager.py` | `plugins/*.py` |

The manager's job is the registry and protocol — not the implementation.

---

## The One Exception

If you have a **tiny, throwaway plugin** used only in a single experiment script and never shared, inlining it into that script is acceptable. But even then, the plugin manager still loads it from its own file — it just happens to live outside the `plugins/` directory.

---

## Summary

```
plugin_manager.py  →  orchestrator only  (discovers, loads, applies)
plugins/*.py       →  implementations    (the actual algorithms)
```

Keep them separate. This is what makes the system extensible by users, testable in isolation, and maintainable over time.
