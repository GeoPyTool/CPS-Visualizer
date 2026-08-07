# Cross-Disciplinary Sample Data

Publicly available LA-ICP-MS datasets from fields **outside** geology,
demonstrating that CPS-Visualizer applies to any scenario where spatially
resolved compositional data benefit from visual comparison.

## 1. Bivalve Shell Trace-Element Profiles — environmental / marine biology

Growth-line element/Ca transects of cockle (*Cerastoderma edule*) and Manila
clam (*Ruditapes philippinarium*), measured by LA-ICP-MS.

- **Source:** de Winter, N. (2025). *Trace element profiles and environmental
  data for cultured Cerastoderma edule and Ruditapes philippinarium.* Zenodo.
  https://doi.org/10.5281/zenodo.18873283  (CC-BY 4.0)
- **Files used:** `B325_int.csv` (cockle), `G668_sub.csv` (clam)
- **Isotopes:** Ca-43 (cps), Na/Mg/Mn/Sr/Ba normalised to Ca (mmol/mol)
- **Format here:** each isotope series reshaped into a 2-D waterfall matrix
  (rows = segments along the shell growth axis, columns = within-segment
  window) — directly loadable into CPS-Visualizer as a surface-scan image.

Files in `bivalve_shell/`:

| File        | Species / sample | Size    | Content            |
|-------------|------------------|---------|--------------------|
| B325Ca.csv  | cockle           | 160×168 | Ca-43 CPS          |
| B325Na.csv  | cockle           | 160×168 | Na/Ca (mmol/mol)   |
| B325Mg.csv  | cockle           | 160×168 | Mg/Ca (mmol/mol)   |
| B325Mn.csv  | cockle           | 160×168 | Mn/Ca (mmol/mol)   |
| B325Sr.csv  | cockle           | 160×168 | Sr/Ca (mmol/mol)   |
| B325Ba.csv  | cockle           | 160×168 | Ba/Ca (mmol/mol)   |
| G668Ca.csv  | clam             | 160×144 | Ca-43 CPS          |
| G668Na.csv  | clam             | 160×144 | Na/Ca (mmol/mol)   |
| G668Mg.csv  | clam             | 160×144 | Mg/Ca (mmol/mol)   |
| G668Mn.csv  | clam             | 160×144 | Mn/Ca (mmol/mol)   |
| G668Sr.csv  | clam             | 160×144 | Sr/Ca (mmol/mol)   |
| G668Ba.csv  | clam             | 160×144 | Ba/Ca (mmol/mol)   |

## 2. Hair Mercury LA-ICP-MS Tracks — biomedical / archaeometry

Mercury exposure study of 19th-century New Zealand goldminers.

- **Source:** King, C. (2021). *DrCharlieKing/Hair_Hg: First release raw data.*
  Zenodo. https://doi.org/10.5281/zenodo.5156997  (CC-BY 4.0)
- **Note:** the raw Iolite export is a spot-calibration table; tracks were
  too short to form meaningful 2-D maps and are therefore not packaged as
  sample matrices. The citation is retained as a biomedical application
  reference.

## How to use

1. Start the web interface:
   ```bash
   python -c "import cpsvisualizer; cpsvisualizer.web()"
   ```
2. Drag the `*.csv` files from `bivalve_shell/` into the drop zone.
3. Compare Ca vs Sr vs Ba maps with the wipe tool, or run the distance /
   statistics / AODA analyses.

## Reproducibility

The matrices are generated from the original Zenodo exports by:

```bash
python crossdisciplinary/convert_sample_data.py
```
