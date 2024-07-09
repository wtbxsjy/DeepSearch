import pandas as pd 
import itertools
from copy import deepcopy
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_digested_db.py <input_file.txt> <output_file.fasta>")
        exit(1)
    
    df = pd.read_csv(
    str(sys.argv[1]), delimiter='\t')
    df = df[df['Sequence'].apply(lambda x: len(x) >= 7 and len(x) <= 32)]
    df = df[df['Sequence'].apply(lambda x: 'X' not in x)]
    df = df[df['Sequence'].apply(lambda x: 'Z' not in x)]
    df = df[df['Sequence'].apply(lambda x: 'B' not in x)]
    df = df[df['Sequence'].apply(lambda x: 'J' not in x)]


    seq2prot = {}
    for _, row in df.iterrows():
        seq = row['Sequence']
        protein = row['Protein_Name']
        is_decoy = protein.startswith('XXX_')

        uni_seq = seq.replace('L', 'I')

        # decoy sequence is at the end of the file
        if uni_seq not in seq2prot:
            seq2prot[uni_seq] = set()
            seq2prot[uni_seq].add(protein)

        else:
            if is_decoy:
                if not any(map(lambda x: x.startswith('XXX_'), seq2prot[uni_seq])):
                    # this seq is not decoy 
                    continue
                else: 
                    seq2prot[uni_seq].add(protein)
            else:
                seq2prot[uni_seq].add(protein)
    
    df = df.drop_duplicates(subset=['Sequence'])
    with open(str(sys.argv[2]), 'w') as f:
        for _, row in df.iterrows():
            seq = row['Sequence']
            uni_seq = seq.replace('L', 'I')
            proteins = seq2prot[uni_seq]
            f.write('>' + ';'.join(proteins) + '\n')
            f.write(row['Sequence'] + '\n')

if __name__ == '__main__':
    main()
