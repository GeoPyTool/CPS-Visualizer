# Cross-Disciplinary Sample Data

Publicly available LA-ICP-MS datasets from fields **outside** geology,
demonstrating that CPS-Visualizer applies to any scenario where spatially
resolved compositional data benefit from visual comparison.

> **Location:** these matrices are packaged under `../DataSample/` so that all
> bundled sample suites live together in the repository's `DataSample/`
> directory. This folder keeps the conversion scripts and full citation record.

## Suite 1 — Bivalve Shell (marine biology / environmental)

Growth-line element/Ca transects of cockle (*Cerastoderma edule*) and Manila
clam (*Ruditapes philippinarium*).

- **Source:** de Winter, N. (2025). *Trace element profiles and environmental
  data for cultured Cerastoderma edule and Ruditapes philippinarium.* Zenodo.
  https://doi.org/10.5281/zenodo.18873283  (CC-BY 4.0)
- **Files:** `../DataSample/BivalveShell_MarineBiology/` — Ca-43 (cps) and
  Na/Mg/Mn/Sr/Ba normalised to Ca (mmol/mol), reshaped to 160×168 / 160×144
  waterfall matrices.

## Suite 2 — Medieval Ceramics (archaeology)

LA-ICP-MS trace-element composition of 14 medieval ceramic samples (8 fabric
groups) from the Madayi site, Indian Ocean trade.

- **Source:** Petrik, J., Slavíček, K., Modérado, M., et al. (2025). *Ceramic
  Trade and Transformation in Medieval Madayi.* Zenodo.
  https://doi.org/10.5281/zenodo.16995032  (CC-BY 4.0)
- **Files:** `../DataSample/Ceramics_Archaeology/` — `Ceramics_TraceElements.csv`
  (14 samples × 44 elements, µg/g), `elements.txt` (element list).

## Suite 3 — Tissue Imaging (biomedical)

LA-ICP-MS imaging of prostate-tissue sections: a 200×219 intensity grid for
P-31, Fe-57, Zn-64 and Zn-66.

- **Source:** Buchholz, R., Krossa, S., Andersen, M. K., et al. (2022). A simple
  preparation protocol for shipping and storage of tissue sections for
  LA-ICP-MS imaging. *Metallomics* 14, mfac013.
  Zenodo. https://doi.org/10.5281/zenodo.6204296  (CC-BY 4.0)
- **Files:** `../DataSample/Tissue_Biomedical/` — `Tissue_P31.csv`,
  `Tissue_Fe57.csv`, `Tissue_Zn64.csv`, `Tissue_Zn66.csv` (each 200×219).

## How to use

1. Start the web interface:
   ```bash
   python -c "import cpsvisualizer; cpsvisualizer.web()"
   ```
2. Use the **Load Sample** menu to load any suite, or drag the `*.csv` files
   directly into the drop zone.
3. Compare isotopes / elements / fabrics with the wipe tool, or run the
   distance / statistics / AODA analyses.

## Reproducibility

The matrices are generated from the original archives by:

```bash
python crossdisciplinary/convert_sample_data.py   # bivalve shells
python crossdisciplinary/convert_more_samples.py  # ceramics + tissue
```
