# DeepSearch

## End-to-end Database Search for Tandem Mass Spectrometry

DeepSearch is a deep learning-based tool for peptide identification from tandem mass spectrometry data. The model uses a modified transformer-based encoder-decoder architecture. DeepSearch is trained under the contrastive learning framework coupled with a *de novo* sequencing task. DeepSearch learns a joint embedding space across the modality of peptide sequence and tandem mass spectra. 

For more details, see our preprint at https://arxiv.org/abs/2405.06511v1.

This software is free to use for non-commercial purpose. 

## Setup DeepSearch

DeepSearch is implemented and tested with Python 3.10, pytorch 2.0 and cuda 12.0 on Linux. To setup DeepSearch, we recommend creating a python virtual environment first. You can find the documentation for creating virtual environment [here](https://docs.python.org/3/library/venv.html), or use a tool like [pyenv](https://github.com/pyenv/pyenv). Anaconda is currently not supported. Please make sure pytorch and cuda is properly installed, and we recommend using a GPU device. DeepSearch depends on the following packages:

- pandas
- numpy
- scipy
- torch
- pyyaml
- h5py
- einops
- tensorboard
- pyteomics
- tqdm

To install DeepSearch, in the project root directory, type:
```
python ./setup.py install
```

## Use DeepSearch
DeepSearch is a resolution-free database search tool. Currently, we support only HCD spectra, and DeepSearch works best with spectra from Thermo Fisher platform. Current version requires preprocessed MS/MS spectra and digested peptide database. The memory requirement for DeepSearch depends on the size of the database. For reference, you need at least 20GB of RAM for human protein database. For reference, DeepSearch finishes within 15 mins for ~40,000 spectra when search against precomputed human peptide embedding database on an A100 GPU.

#### Prepare your data
We recommend [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser) for preprocessing raw spectra file into mzML/mgf format:
```
ThermoRawFileParser.exe -d /input/directory  -o /output/directory -f=0 -p=2
```
Protein database need to be pre-digested. For best practice, we recommend using UniProtKB with isoform sequence and up to 2 missed cleavage. Current version support peptide sequences up to 32 amino acids. To replicate paper result, we recommend [Protein Digestion Simulator](https://github.com/PNNL-Comp-Mass-Spec/Protein-Digestion-Simulator). An example script is included for processing the digested file (ignore IL difference and combine duplicate sequences), see scripts/parse_digested_db.py. 

#### Search digested database
We recommend using a precursor p.p.m of +/- 10. Notice that DeepSearch will compute an embedding database for peptide sequence for the first time. The embedding database can be reused for further search. See available parameters for DeepSearch:
```
python ./bin/search.py --help
```

#### Running demo file
The provided demo files includes a MS2 spectra file in mgf format and a pre-digested uniprot database for E. coli. After properly setup the environment: 

```
python ./bin/search.py --input ./demo/ecoli_demo.mgf \
                        --model path_to_param.pt \
                        --database ./demo/uniprot_ecoli_tryptic_digested.fasta \
                        --config ./config/CTandem_mini.yaml \
                        --device='cuda' \
                        --precursor-ppm 10 \ 
                        --decoy-prefix XXX_ \
                        --min-pep-len 7 \
                        --compute-embedding \
                        --reselection \
                        --result-dir ./demo/result 
```

Notice that you need to specify --compute-embedding if you want to reuse the peptide embedding database. Otherwise search will be perform on the fly and may consume more time for larger database. The computed peptide embedding database will be stored under the result directory in npy format. To reuse, simply specify --embedding ./path/to/embedding.npy.

Adjust the param --peptide-batch-size and --spectra-batch-size based on your hardward. For a GPU with 8GB memory, we recommand 256 for peptide batch size and 64 for spectra batch size. 

#### Interpret search result
DeepSearch outputs search result in tsv, including the following information:

- file: the spectra file
- scan: spectra scan number
- pre_mz: precursor m/z of spectra
- charge: precursor charge of spectra
- rt: retention time of spectra
- peptide: matched peptide sequence
- modified_peptide: peptide sequence with its modification profile
- neutral_mass: calculated neutral mass from the matched sequence
- ppm_error: estimated ppm error for the match
- score: cosine similarity score for the match
- protein: the protein identifier for the matched peptide
- decoy: whether the match is a decoy or not

The combined result and FDR filtered result is stored as all_PSM.tsv and controlled_PSM.tsv correspondingly under the result folder.

## Train your own model
DeepSearch is trained on the [MassIVE-KB](https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp) *in vivo* HCD dataset. See an example [here](https://github.com/bittremieux/GLEAMS) for downloading the dataset. To create file for training, please use the provided script build_dataset_IVE_con.py. Alternatively, the already processed training data is available among request. To see the training parameters:
```
python ./bin/train_model.py --help
```
DeepSearch is trained with a batch size of 15,360 using gradient accumulation. Please modify the batch size based on your hardware. We recommend distributed training on clusters. For reference, DeepSearch was trained with 8 A100-SXM 40GB GPUs for 10 epochs. 

