# DISMaL: Demographic Inference under the Structured coalescent using Maximum Likelihood

Manual for user interface (not documentation for current version)

## Contents
1. Installation
2. How to use DISMaL
3. Structured coalescent and how DISMaL works

## 1. Installation
Install via bioconda: `conda install -c bioconda dismal`.

## 2. How to use DISMaL

### 2a: Input file format
The input for a DISMaL analysis is a FASTA file in a modified DILS format. Each sequence is a block of an arbitrary but consistent length. It is assumed that there is free recombination between blocks, and negligble recombination within blocks.

Headers follow this format:
```
> block id|species or population|individual|allele or haplotype id
```

For haploids, or where phase information is not known, the allele id can be arbitrary, but not left blank.

Users must use their biological judgement in deciding on block length. Blocks need to be long enough to provide sufficient numbers of segregating sites, so species with lower mutation rates need longer blocks. Recombination rates also complicate this; species with high recombination rates may need shorter blocks to ensure that recombination within blocks is negligble, and species with low recombination rates will need larger gaps between blocks to ensure pseudo-free recombination between them.

### 2b: Specifying a single model in DISMaL
The central workhorse of DISMaL is the `DemographicModel` class. The default `DemographicModel` is a simple 2-epoch isolation model, in which one species diverges at time tau and forms two populations, without any gene flow.
```
from dismal.inference import DemographicModel
mod = DemographicModel() # 2 epoch model, no migration
```
Adding an epoch always allows for population size changes, so if we want to fit a model with three epochs (one pre-divergence, two-post divergence without migration) in which the population sizes change at some unspecified time tau1:
```
mod.add_epoch(t_var_name = "tau1")
```

Alternatively, we can allow symmetric migration during that epoch:
```
mod.add_epoch(t_var_name = "tau1", migration=True)
```

Or we can allow asymmetric migration:
```
mod.add_epoch(t_var_name = "tau1", migration=True, asymmetric_migration=True)
```

Or we can specify unidirectional migration, specifying the movement of genes **backwards in time**:
```
mod.add_epoch(t_var_name = "tau1", migration=True, migration_direction=[SOURCE, DEST])
```
Where SOURCE and DEST correspond to the species/population names in the input file.

We can then add another epoch if we so wish:
```
mod.add_epoch(t_var_name = "tau2", migration=True, migration_direction=[DEST, SOURCE])
```

### 2c: Fitting a single model
Once the correct model has been specified, inference is easy:
```
res = mod.infer(data=FILE)
```

### 2d: Fitting and comparing multiple models
Often, demographic models are only useful if we can compare multiple competing models and select the best fit. DISMaL aims to make this easy, fast, and rigorous. This _can_ be done manually, for instance using `DemographicModel().aic()`, but the `MultiModel` class provides functionality to automatically fit and compare several DISMaL models with little code.

By default, DISMaL will fit up to 14 models (in this order):
1. 2-epoch isolation model (4 parameters)
2. 2-epoch symmetric IM model (5 parameters)
3. 2-epoch asymmetric IM
4. 3-epoch isolation model
5. 3-epoch single migration rate model
6. 3-epoch IIM model
7. 3-epoch secondary contact model
8. etc...


DISMaL will work its way through each model and compare them, until it finds a model that performs better than a) all simpler models, and b) all models with the same number of parameters.
Finally, DISMaL performs a likelihood ratio test between the inferred model and either the 2-epoch or 3-epoch isolation model (depending on whether the chosen model is 2-epoch).

The `MultiModel` interface works like this:
```
multimod = MultiModel(max_epochs = 3, data = FILE)
best_mod = multimod.best_model
```
