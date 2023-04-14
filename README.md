# DISMaL: *D*emographic *I*nference under the *S*tructured Coalescent using *Ma*ximum *L*ikelihoods

## Installation

First set up the provided conda environment:
```
conda env create -f dismal.yml
conda activate dismal
```

Then install DISMaL into the conda environment:
```
pip install -e .
```

## Preprocess data
There are three methods to read in and preprocess data in DISMaL:
1. Provide a single VCF and let DISMaL create blocks
2. Provide a directory of VCFs, each corresponding to a block
3. Providing raw counts of nucleotide differences

### Files required for VCF processing
Options (1) and (2) require a comma-separated mapping file from sample names to population names:
```
john,human
lisa,human
skittles,cat
snowball,cat
```
Population names are arbitrary and do not matter outside of VCF processing.

### Option 1: Providing a single VCF
DISMaL currently provides (extremely) naive blocking functionality, making the assumption that all parts of the genome should be partitioned into blocks, and that there are no missing data:
```
from dismal import preprocess
s_counts = preprocess.vcf_to_s_count(vcf_path=[VCF_PATH], samples_path=[SAMPLES_MAP_PATH], block_length=64, n_blocks=10000)
```
Notes:
* `SAMPLES_MAP_PATH` is the path to the sample-to-population mapping file mentioned above. 
* The genome must be minimally of size `block_length * n_blocks`
* Currently, DISMaL (incorrectly) assumes that the position of the last variant is the length of the genome
* `n_blocks` is the number of blocks randomly sampled from the genome

### Option 2: Multiple VCFs, each corresponding to a single block
```
from dismal import preprocess
s_counts = preprocess.block_vcfs_to_s_count(directory=[VCFs_DIRECTORY], samples_path=[SAMPLES_MAP_PATH], file_pattern="*.vcf")
```
`block_vcfs_to_s_count` reads all files in `VCFs_DIRECTORY` that match the `file_pattern` parameter, and combine the output into a single dictionary of counts of nucleotide differences per state.

### Option 3: Provide a list of dictionaries of nucleotide differences
This option skips the VCF processing altogether. To fit models, a list of dictionaries is required:
* Each of the three dictionaries in the list correspond to the sampling state
* Each key in each dictionary is the number of nucleotide differences between two blocks
* Each value is the observed frequency of each key

For example:
```
s_counts = [{0:20, 1:16, 2:8, 3:4, 4:2, 5:1}, 
            {0:19, 1:17, 2:7, 3:3, 4:1, 5:3},
            {0:3, 1:5, 2:10, 4:4, 5:6, 6:4, 8:3, 10:1}]
```
## Inference of demographic parameters

Demographic models are fit using the `Demography()` class. Models can be one of the four pre-specified ones:
* Isolation (`model="iso"`)
* Isolation with initial migration (`model="iim"`)
* Secondary contact (`model="sec"`)
* Generalised introgression with migration (`model="gim"`)

```
from dismal.demography import Demography
Demography(X=s_counts, model="iso").infer_parameters()
```

Alternatively, each migration rate can be allowed or disallowed by specifying one or several of:
* `set_m1_zero`
* `set_m2_zero`
* `set_m1_prime_zero`
* `set_m2_prime_zero`

For instance, if we wish to fit a model where the migration rate m1 is zero, but all other rates are allowed to be non-zero:
```
Demography(X=s_counts, set_m1_zero=True).infer_parameters()
```

### Options

The `Demography` class allows the user to specify various parameters:
```
Demography(X, model_description=None, model=None,
                 set_m1_zero = False, set_m2_zero = False, set_m1_prime_zero = False, set_m2_prime_zero = False, no_migration=False,
                 a_initial=1, b_initial=1, c1_initial=1, c2_initial=1, tau0_initial=1, tau1_initial=2, m1_initial=0, m2_initial=0, 
                 m1_prime_initial=0, m2_prime_initial=0, theta_initial=5, 
                 a_lower=0.001, b_lower=0.001, c1_lower=0.001, c2_lower=0.001, tau0_lower=0.001, tau1_lower=0.001, m1_lower=0, m2_lower=0, 
                 m1_prime_lower=0, m2_prime_lower=0, theta_lower=0.0000001)
```
The two sets of parameters not mentioned above that the user can currently specify are initial values and lower bounds. These have both been taken from Costa & Wilkinson-Herbots (2021), with some modification of the lower bounds to allow the user to specify them in the original model parameters.
