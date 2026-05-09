"""
Disease Gene Ground Truth Mapping for Knockout Simulation
=========================================================
50 well-characterized monogenic disease genes mapped to OMNIS departments.
Ground truth defined BEFORE simulation (documented in supplementary table).

Sources: OMIM, ClinVar pathogenic/likely pathogenic, published reviews.
Each mapping: Gene -> Department -> Disease -> Mechanism (brief justification)
"""

DISEASE_GENE_GROUND_TRUTH = {
    "BRCA1":  {"department": "DNA repair",      "disease": "Hereditary breast/ovarian cancer",    "mechanism": "Homologous recombination repair of DSBs"},
    "BRCA2":  {"department": "DNA repair",      "disease": "Hereditary breast/ovarian cancer",    "mechanism": "RAD51-mediated homologous recombination"},
    "TP53":   {"department": "Apoptosis",        "disease": "Li-Fraumeni syndrome",                "mechanism": "Apoptosis/cell cycle checkpoint loss"},
    "RB1":    {"department": "Cell cycle",       "disease": "Retinoblastoma",                      "mechanism": "G1/S cell cycle checkpoint control"},
    "CFTR":   {"department": "Ion channel",      "disease": "Cystic fibrosis",                     "mechanism": "Chloride channel dysfunction"},
    "SCN1A":  {"department": "Ion channel",      "disease": "Dravet syndrome",                     "mechanism": "Voltage-gated sodium channel Nav1.1"},
    "KCNQ1":  {"department": "Ion channel",      "disease": "Long QT syndrome type 1",             "mechanism": "Voltage-gated potassium channel"},
    "SCN5A":  {"department": "Ion channel",      "disease": "Brugada syndrome / Long QT type 3",   "mechanism": "Cardiac sodium channel Nav1.5"},
    "CACNA1A":{"department": "Ion channel",      "disease": "Episodic ataxia type 2 / SCA6",       "mechanism": "P/Q-type calcium channel"},
    "PKD1":   {"department": "Transport",        "disease": "Polycystic kidney disease",            "mechanism": "Polycystin-1 cation channel complex"},
    "PKD2":   {"department": "Transport",        "disease": "Polycystic kidney disease type 2",     "mechanism": "Polycystin-2 TRP cation channel"},
    "SLC12A3":{"department": "Transport",        "disease": "Gitelman syndrome",                    "mechanism": "Thiazide-sensitive NaCl cotransporter"},
    "COL1A1": {"department": "Structural",       "disease": "Osteogenesis imperfecta",              "mechanism": "Type I collagen structural protein"},
    "COL7A1": {"department": "Structural",       "disease": "Dystrophic epidermolysis bullosa",     "mechanism": "Type VII collagen anchoring fibrils"},
    "FBN1":   {"department": "Structural",       "disease": "Marfan syndrome",                      "mechanism": "Fibrillin-1 microfibril scaffold"},
    "DMD":    {"department": "Structural",       "disease": "Duchenne muscular dystrophy",          "mechanism": "Dystrophin cytoskeletal anchor"},
    "LMNA":   {"department": "Structural",       "disease": "Emery-Dreifuss muscular dystrophy",    "mechanism": "Lamin A/C nuclear envelope structure"},
    "FGFR3":  {"department": "Signaling",        "disease": "Achondroplasia",                       "mechanism": "FGF receptor tyrosine kinase signaling"},
    "PTCH1":  {"department": "Signaling",        "disease": "Gorlin syndrome / basal cell nevus",   "mechanism": "Hedgehog pathway receptor"},
    "NOTCH1": {"department": "Signaling",        "disease": "Adams-Oliver syndrome",                "mechanism": "Notch signaling receptor"},
    "NF1":    {"department": "GTPase",           "disease": "Neurofibromatosis type 1",             "mechanism": "Ras-GAP, inactivates RAS GTPase"},
    "NF2":    {"department": "Cytoskeleton",     "disease": "Neurofibromatosis type 2",             "mechanism": "Merlin, links cytoskeleton to membrane"},
    "TSC1":   {"department": "GTPase",           "disease": "Tuberous sclerosis",                   "mechanism": "Hamartin, inhibits mTOR via Rheb GTPase"},
    "TSC2":   {"department": "GTPase",           "disease": "Tuberous sclerosis",                   "mechanism": "Tuberin, GAP for Rheb GTPase"},
    "HEXA":   {"department": "Proteolysis",      "disease": "Tay-Sachs disease",                    "mechanism": "Hexosaminidase A lysosomal degradation"},
    "GBA":    {"department": "Proteolysis",      "disease": "Gaucher disease",                      "mechanism": "Glucocerebrosidase lysosomal hydrolysis"},
    "GAA":    {"department": "Proteolysis",      "disease": "Pompe disease",                        "mechanism": "Acid alpha-glucosidase lysosomal"},
    "IDUA":   {"department": "Proteolysis",      "disease": "Hurler syndrome (MPS I)",              "mechanism": "Alpha-L-iduronidase lysosomal"},
    "PTEN":   {"department": "Phosphatase",      "disease": "Cowden syndrome",                      "mechanism": "PI3K/AKT pathway lipid phosphatase"},
    "PTPN11": {"department": "Phosphatase",      "disease": "Noonan syndrome",                      "mechanism": "SHP2 tyrosine phosphatase"},
    "VHL":    {"department": "Ubiquitin",        "disease": "Von Hippel-Lindau disease",            "mechanism": "E3 ubiquitin ligase substrate recognition"},
    "PARK2":  {"department": "Ubiquitin",        "disease": "Parkinson disease (AR juvenile)",      "mechanism": "Parkin E3 ubiquitin ligase"},
    "UBE3A":  {"department": "Ubiquitin",        "disease": "Angelman syndrome",                    "mechanism": "E3 ubiquitin-protein ligase"},
    "ATM":    {"department": "DNA repair",       "disease": "Ataxia-telangiectasia",                "mechanism": "DSB sensor kinase, DNA damage response"},
    "FANCC":  {"department": "DNA repair",       "disease": "Fanconi anemia comp. C",               "mechanism": "Interstrand crosslink repair"},
    "MLH1":   {"department": "DNA repair",       "disease": "Lynch syndrome (HNPCC)",               "mechanism": "DNA mismatch repair"},
    "APC":    {"department": "Signaling",        "disease": "Familial adenomatous polyposis",       "mechanism": "Wnt/beta-catenin signaling regulation"},
    "CDH1":   {"department": "Cell adhesion",    "disease": "Hereditary diffuse gastric cancer",    "mechanism": "E-cadherin cell-cell adhesion"},
    "HTT":    {"department": "Transport",        "disease": "Huntington disease",                   "mechanism": "Vesicular transport, polyQ aggregation"},
    "SMN1":   {"department": "RNA processing",   "disease": "Spinal muscular atrophy",              "mechanism": "snRNP assembly, pre-mRNA splicing"},
    "DYNC1H1":{"department": "Transport",        "disease": "Charcot-Marie-Tooth type 2O",          "mechanism": "Cytoplasmic dynein heavy chain transport"},
    "FOXP2":  {"department": "Transcription",    "disease": "Speech-language disorder",             "mechanism": "Forkhead box transcription factor"},
    "PAX6":   {"department": "Transcription",    "disease": "Aniridia",                             "mechanism": "Paired box transcription factor"},
    "SOX9":   {"department": "Transcription",    "disease": "Campomelic dysplasia",                 "mechanism": "SRY-box transcription factor"},
    "RUNX2":  {"department": "Transcription",    "disease": "Cleidocranial dysplasia",              "mechanism": "Runt-related transcription factor"},
    "CREBBP": {"department": "Chromatin",        "disease": "Rubinstein-Taybi syndrome",            "mechanism": "CBP histone acetyltransferase"},
    "ATRX":   {"department": "Chromatin",        "disease": "Alpha-thalassemia X-linked ID",       "mechanism": "SWI/SNF chromatin remodeler"},
    "DNMT3B": {"department": "Methylation",      "disease": "ICF syndrome",                         "mechanism": "De novo DNA methyltransferase"},
    "WAS":    {"department": "Immune",           "disease": "Wiskott-Aldrich syndrome",             "mechanism": "Actin cytoskeleton in immune cells"},
    "JAK3":   {"department": "Kinase",           "disease": "Severe combined immunodeficiency",     "mechanism": "Janus kinase cytokine signaling"},
}

DISEASE_GENES_NEEDING_REVIEW = []

assert len(DISEASE_GENE_GROUND_TRUTH) == 50, f"Expected 50 disease genes, got {len(DISEASE_GENE_GROUND_TRUTH)}"

dept_coverage = {}
for gene, info in DISEASE_GENE_GROUND_TRUTH.items():
    dept = info["department"]
    dept_coverage[dept] = dept_coverage.get(dept, 0) + 1

if __name__ == "__main__":
    print(f"Disease gene ground truth: {len(DISEASE_GENE_GROUND_TRUTH)} genes")
    print(f"\nDepartment coverage ({len(dept_coverage)}/22 departments):")
    for dept in sorted(dept_coverage, key=lambda d: -dept_coverage[d]):
        genes = [g for g, i in DISEASE_GENE_GROUND_TRUTH.items() if i["department"] == dept]
        print(f"  {dept:20s}: {dept_coverage[dept]:2d}  ({', '.join(genes)})")
    print(f"\nDepartments not covered: {22 - len(dept_coverage)}")
    uncovered = set(['Apoptosis','Cell adhesion','Cell cycle','Chromatin','Cytoskeleton',
                     'DNA repair','GTPase','Immune','Ion channel','Kinase','Methylation',
                     'Nuc acid bind','Phosphatase','Protein folding','Proteolysis',
                     'RNA processing','Signaling','Structural','Transcription',
                     'Translation','Transport','Ubiquitin']) - set(dept_coverage.keys())
    for d in sorted(uncovered):
        print(f"  {d}")
