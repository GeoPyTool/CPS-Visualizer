# CPS-Visualizer Sample Data

Bundled example LA-ICP-MS datasets for trying the tool without your own data.

## Layout

```
DataSample/
├── Geology/                          # geological element surface scans (LA-ICP-MS)
│   ├── Ag.csv   Au.csv   Cu.csv      # 126 × 1991 element-distribution matrices
│   ├── Fe.csv   Pb.csv   Zn.csv
├── BivalveShell_MarineBiology/       # marine biology / environmental LA-ICP-MS
│   ├── B325Ca.csv … B325Ba.csv       # cockle (Cerastoderma edule) element/Ca, 160×168
│   ├── G668Ca.csv … G668Ba.csv       # Manila clam (Ruditapes philippinarium), 160×144
├── Ceramics_Archaeology/             # archaeology LA-ICP-MS
│   ├── Ceramics_TraceElements.csv    # 14 medieval ceramic samples × 44 elements (µg/g)
│   └── elements.txt
└── Tissue_Biomedical/                # biomedical LA-ICP-MS imaging
    ├── Tissue_P31.csv  Tissue_Fe57.csv   # 200 × 219 tissue-section intensity grids
    ├── Tissue_Zn64.csv Tissue_Zn66.csv
```

Each subfolder carries its own README with the dataset source and citation.

## Usage

1. Start the web interface:
   ```bash
   python -c "import cpsvisualizer; cpsvisualizer.web()"
   ```
2. Use the **Load Sample** menu in the header to load any suite, or drag the
   CSV files directly into the drop zone.
3. Explore the Map Viewer, run distances, statistics, or AODA.

## Sources

- **Geology:** extracted from `DataSample.zip` (Ag/Cu/Fe/Zn) plus Au and Pb from
  the authors' published study (Yu, 2019). Full citation in the manuscript.
- **Marine biology:** de Winter, N. (2025). Zenodo.
  https://doi.org/10.5281/zenodo.18873283 (CC-BY 4.0).
- **Archaeology:** Petrik, J. et al. (2025). Zenodo.
  https://doi.org/10.5281/zenodo.16995032 (CC-BY 4.0).
- **Biomedical:** Buchholz, R. et al. (2022). *Metallomics* 14, mfac013.
  Zenodo. https://doi.org/10.5281/zenodo.6204296 (CC-BY 4.0).
