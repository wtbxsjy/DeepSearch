import argparse


fix_modification_id = {
    1: 'CAM',
}

var_modification_id = {
    1: 'OX', # Oxidation
    2: 'PHOS', # Phosphorylation
    3: 'ACET', # Nterm - Acetylation
}


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="Path to mgf formated spectra file, specify directory path if multiple files.",
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model params, in ckpt format.",
        required=True
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to model config, in yaml format.",
        default='../config/DeepSearch_mini.yaml',
        required=True
    )

    parser.add_argument(
        "--database",
        type=str,
        help="Path to digested peptide database, in fasta format.",
        required=True
    )

    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Path to precomputed peptide embedding database, in npy format. If not specified, search will be performed on the fly.",
        required=False
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of workers when loading dataset, default=1."
    )

    parser.add_argument(
        "--compute-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify to compute embedding database before search, computed embedding database will be stored as ./result-dir/database_name.pkl.",
        required=False
    )

    parser.add_argument(
        "--reselection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify to use de novo module for PSM reselection.",
        required=False
    )

    parser.add_argument(
        "--in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify to use in memory computation.",
        required=False
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./result",
        help="Directory for storing output."
    )

    parser.add_argument(
        "--decoy-prefix",
        type=str,
        default='DECOY_',
        help="Prefix for decoy sequences"
    )

    parser.add_argument(
        "--FDR",
        type=float,
        default=0.01,
        help="FDR control for output, set to 1 for no FDR."
    )

    # search params
    parser.add_argument(
        "--precursor-ppm",
        type=int,
        default=10,
        help="Precursor mass tolerence, in ppm."
    )

    parser.add_argument(
        "--max-pep-len",
        type=int,
        default=32,
        help="Maximum peptide length, default=32."
    )

    parser.add_argument(
        "--min-pep-len",
        type=int,
        default=6,
        help="Minimum peptide length, default=6."
    )

    parser.add_argument(
        "--min-charge",
        type=int,
        default=2,
        help="Minimum precursor charge, default=2."
    )

    parser.add_argument(
        "--max-charge",
        type=int,
        default=4,
        help="Maximum precursor charge, default=4."
    )

    parser.add_argument(
        "--min-peaks",
        type=int,
        default=64,
        help="Minimum number of peaks in spectra, default=64."
    )

    parser.add_argument(
        "--min-mz",
        type=float,
        default=0.,
        help="Minimum m/z to consider, default=0."
    )

    parser.add_argument(
        "--max-mz",
        type=float,
        default=2000.,
        help="Maximum m/z to consider, default=2000."
    )

    # PTM params
    parser.add_argument(
        "--fix-mod",
        nargs='+',
        type=int,
        default=1,
        help="List of fixed modification, default=1 (Carbamidomethyl on Cysteine)",
    )

    parser.add_argument(
        "--var-mod",
        nargs='+',
        type=int,
        help="List of variable modification, default=None. 1: Oxidation, 2: Phosphorylation, 3: Nterm - Acetylation",
    )

    parser.add_argument(
        "--max-mod",
        type=int,
        default=3,        
        help="Maximum number of variable modification in peptide, default=3",
    )
    
    # device allocation params
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',        
        help="Device used for tensor operation, default=cpu",
    )

    parser.add_argument(
        "--open-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify to open search.",
        required=False
    )


    parser.add_argument(
        "--open-search-mass-tol",
        type=float,
        default=350,
        help="Open search mass tolerance, in Dalton.",
    )


    parser.add_argument(
        "--spectra-batch-size",
        type=int,
        default=256,
        help="Maximum number of spectra processed in the same time, adjust based on GPU/CPU memory",
    )
    
    parser.add_argument(
        "--peptide-batch-size",
        type=int,
        default=256,
        help="Maximum number of peptide processed in the same time, adjust based on GPU/CPU memory",
    )


    args = parser.parse_args(args)

    return args
