import pandas as pd
import numpy as np

def get_peptide_pep(hits: pd.DataFrame):
    peptide2pep = {}
    for i, row in hits.iterrows():
        peptide = row['peptide']
        pep = row['pep']
        if peptide not in peptide2pep:
            peptide2pep[peptide] = 1000
        peptide2pep[peptide] = min(pep, peptide2pep[peptide])
    return peptide2pep


def build_protein_group(hits: pd.DataFrame):
    proteins = set()
    protein2peptides = {}
    peptide2proteins = {}
    peptide2pep = get_peptide_pep(hits)
    protein2pep = {}
    degenerate_peptides = set()

    for i, row in hits.iterrows():
        protein = row['protein']
        peptide = row['peptide']
        pep = peptide2pep[peptide]
        protein = protein.split(';')
        for p in protein:
            proteins.add(p)
            if p not in protein2peptides:
                protein2peptides[p] = set()
                protein2pep[p] = 1000
            protein2pep[p] = min(pep, protein2pep[p])
            protein2peptides[p].add(peptide)
            if peptide not in peptide2proteins:
                peptide2proteins[peptide] = set()
            else:
                degenerate_peptides.add(peptide)
            peptide2proteins[peptide].add(p)

    l_proteins = list(proteins)
    n_proteins = len(l_proteins)

    aj_list = [[] for _ in range(n_proteins)]

    for i in range(n_proteins):
        for j in range(i+1, n_proteins):
            p1 = l_proteins[i]
            p2 = l_proteins[j]

            if protein2peptides[p1].issubset(protein2peptides[p2]) or protein2peptides[p2].issubset(protein2peptides[p1]):
                aj_list[i].append(j)
                aj_list[j].append(i)

    def connected_components(graph):
        seen = set()
        for root in range(len(graph)):
            if root not in seen:
                seen.add(root)
                component = []
                queue = deque([root])

                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in graph[node]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
                yield component

    protein_groups_idx = list(connected_components(aj_list))
    protein2group = {}
    peptide2group = {}
    group2peptides = {}
    for i, group in enumerate(protein_groups_idx):
        for p in group:
            protein2group[l_proteins[p]] = i
            for peptide in protein2peptides[l_proteins[p]]:
                if peptide not in peptide2group:
                    peptide2group[peptide] = set()
                peptide2group[peptide].add(i)
                if i not in group2peptides:
                    group2peptides[i] = set()
                group2peptides[i].add(peptide)

    degenerate_peptides_ = []
    dpeptides2groups = {}
    for peptide in degenerate_peptides:
        proteins_ = peptide2proteins[peptide]
        groups = set()
        for p in proteins_:
            groups.add(protein2group[p])
        if len(groups) > 1:
            degenerate_peptides_.append(peptide)
            dpeptides2groups[peptide] = groups
    degenerate_peptides = degenerate_peptides_

    for peptide in degenerate_peptides:
        groups = dpeptides2groups[peptide]
        max_n_peptides = 0
        for group in groups:
            if len(group2peptides[group]) > max_n_peptides:
                max_n_peptides = len(group2peptides[group])
                max_group = group
        for group in groups:
            if group != max_group:
                group2peptides[group].remove(peptide)
                peptide2group[peptide].remove(group)

    protein_groups = []
    grouppep = []
    for i, group in enumerate(protein_groups_idx):
        proteins_ = [l_proteins[p] for p in group]
        protein_groups.append(proteins_)
        min_group_pep = 100000
        for peptide in group2peptides[i]:
            if peptide2pep[peptide] < min_group_pep:
                min_group_pep = peptide2pep[peptide]
        grouppep.append(min_group_pep)

    return protein_groups, grouppep
