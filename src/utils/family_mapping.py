"""
GPCR Protein Family Mapping.

Canonical source of truth for receptor -> family assignments.
Import this module in any script that needs family labels.

Usage:
    from family_mapping import assign_family, FAMILY_LOOKUP

    family = assign_family("Adenosine_receptor_A2a~3EML")  # -> "Adenosine"

    #Apply to a DataFrame:
    import pandas as pd
    df = pd.read_csv("some_data.csv")
    df["family"] = df["receptor"].apply(assign_family)
"""

# ══════════════════════════════════════════════════════════════════════════════
# Family -> receptor prefix mapping
# ══════════════════════════════════════════════════════════════════════════════

FAMILIES = {
    "Adenosine": [
        "Adenosine_receptor_A1",
        "Adenosine_receptor_A2a",
    ],
    "Adrenergic": [
        "ADRB2",
        "Beta-1_adrenergic_receptor",
        "Beta-2_adrenergic_receptor",
    ],
    "Peptide": [
        "Corticotropin-releasing_factor_receptor_1",
        "Neurotensin_receptor_type_1",
        "Proteinase-activated_receptor_1",
        "F2R",
        "Endothelin_B_receptor",
        "Endothelin_receptor_type_B",
        "Type-1_angiotensin_II_receptor",
        "Type-2_angiotensin_II_receptor",
    ],
    "Chemokine": [
        "C-C_chemokine_receptor_type_5",
        "C-X-C_chemokine_receptor_type_4",
        "C5a_anaphylatoxin_chemotactic_receptor_1",
        "G-protein_coupled_receptor_homolog_US28",
    ],
    "Muscarinic": [
        "CHRM2",
        "Muscarinic_acetylcholine_receptor_M1",
        "Muscarinic_acetylcholine_receptor_M2",
        "Muscarinic_acetylcholine_receptor_M3",
        "Muscarinic_acetylcholine_receptor_M4",
    ],
    "Cannabinoid": [
        "CNR1",
        "Cannabinoid_receptor_1",
    ],
    "Dopamine": [
        "DRD3",
        "D(3)_dopamine_receptor",
        "D(4)_dopamine_receptor",
    ],
    "Opioid": [
        "Mu-type_opioid_receptor",
        "Delta-type_opioid_receptor",
        "Kappa-type_opioid_receptor",
        "Nociceptin_receptor",
    ],
    "Serotonin": [
        "5-hydroxytryptamine_receptor_1B",
        "5-hydroxytryptamine_receptor_2B",
    ],
    "Histamine": [
        "HRH1",
        "Histamine_H1_receptor",
    ],
    "Visual": [
        "Rhodopsin",
    ],
    "Lipid": [
        "Sphingosine_1-phosphate_receptor_1",
        "Lysophosphatidic_acid_receptor_1",
        "Cysteinyl_leukotriene_receptor_2",
        "PAFR",
    ],
    "Secretin": [
        "Glucagon_receptor",
        "Glucagon-like_peptide_1_receptor",
    ],
    "Purinergic": [
        "P2Y_purinoceptor_1",
        "P2Y_purinoceptor_12",
    ],
    "Melatonin": [
        "Melatonin_receptor_type_1A",
    ],
    "Orexin": [
        "Orexin_receptor_type_1",
        "Orexin_receptor_type_2",
        "Hypocretin_receptor_type_1",
    ],
    "Neuropeptide_Y": [
        "Neuropeptide_Y_receptor_type_1",
    ],
    "Prostaglandin": [
        "Prostaglandin_E2_receptor_EP3_subtype",
    ],
    "Free_Fatty_Acid": [
        "Free_fatty_acid_receptor_1",
    ],
    "Metabotropic_Glutamate": [
        "Metabotropic_glutamate_receptor_1",
        "Metabotropic_glutamate_receptor_5",
    ],
    "Smoothened": [
        "Protein_smoothened",
        "Smoothened_homolog",
    ],
    "Arrestin": [
        "Beta-arrestin-1",
        "Beta-arrestin-2",
    ],
    "G_Protein": [
        "Guanine_nucleotide-binding_protein_G(s)_subunit_alpha_isoforms_short",
    ],
}

# PDB-based overrides for entries with "unknown" receptor names.
# Verified manually against RCSB PDB.
PDB_OVERRIDES = {
    "3P2D": "Arrestin",           # Arrestin-3 (beta-arrestin-2)
    "6K3F": "Arrestin",           # Beta-arrestin-2 + CXCR7 phosphopeptide
    "6KPF": "Cannabinoid",        # Cannabinoid receptor
    "6NBF": "G_Protein",          # G(s) subunit alpha
}

# PDB IDs that are NOT GPCRs and should be excluded.
EXCLUDE_PDB = {
    "3W5A",  # SERCA calcium pump — not a GPCR
}

# ══════════════════════════════════════════════════════════════════════════════
# Build flat lookup: prefix -> family
# ══════════════════════════════════════════════════════════════════════════════

FAMILY_LOOKUP = {}
for family_name, prefixes in FAMILIES.items():
    for prefix in prefixes:
        FAMILY_LOOKUP[prefix] = family_name


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def assign_family(receptor: str) -> str | None:
    """
    Map a receptor string (e.g. 'Adenosine_receptor_A2a~3EML') to its family.

    Returns:
        Family name (str), or None if the receptor should be excluded
        (non-GPCR), or "Other" if no mapping is found.
    """
    # Check exclusion list first
    for pdb in EXCLUDE_PDB:
        if pdb in receptor:
            return None

    # Try prefix matching against known receptor names
    for prefix, family in FAMILY_LOOKUP.items():
        if receptor.startswith(prefix):
            return family

    # Try PDB-based matching for unknown receptors
    for pdb, family in PDB_OVERRIDES.items():
        if pdb in receptor:
            return family

    return "Other"


def get_all_families() -> list[str]:
    """Return sorted list of all family names."""
    return sorted(FAMILIES.keys())


def get_family_receptors(family: str) -> list[str]:
    """Return the receptor prefixes for a given family."""
    return FAMILIES.get(family, [])


# ══════════════════════════════════════════════════════════════════════════════
# CLI: apply mapping to a CSV
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Apply family mapping to a dataset CSV"
    )
    parser.add_argument("input_csv", help="Input CSV with 'receptor' column")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: adds _families suffix)")
    parser.add_argument("--receptor_col", default="receptor",
                        help="Name of the receptor column")
    parser.add_argument("--exclude_non_gpcr", action="store_true", default=True,
                        help="Remove non-GPCR entries (default: True)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["family"] = df[args.receptor_col].apply(assign_family)

    n_excluded = df["family"].isna().sum()
    n_other = (df["family"] == "Other").sum()

    if args.exclude_non_gpcr:
        df = df[df["family"].notna()]

    out_path = args.output or args.input_csv.replace(".csv", "_families.csv")
    df.to_csv(out_path, index=False)

    print(f"Mapped {len(df)} rows to {df['family'].nunique()} families")
    print(f"Excluded {n_excluded} non-GPCR entries")
    if n_other > 0:
        others = df[df["family"] == "Other"][args.receptor_col].unique()
        print(f"WARNING: {n_other} entries mapped to 'Other': {list(others)}")
    print(f"Saved: {out_path}")