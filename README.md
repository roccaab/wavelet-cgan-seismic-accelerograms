# wavelet-cgan-seismic-accelerograms
PyTorch implementation of a Wavelet-Decomposed conditional GAN for synthetic seismic accelerogram generation using DWT sub-band discriminators and energy-weighted adversarial training.

# WD-cGAN for Synthetic Seismic Accelerogram Generation

PyTorch implementation of a **Wavelet-Decomposed conditional Generative Adversarial Network (WD-cGAN)** for the generation of synthetic seismic accelerograms.

This repository accompanies the manuscript:

> **Synthetic Seismic Accelerogram Generation via Wavelet-Decomposed Conditional Generative Adversarial Networks**  
> **Antonio Rocca, Luigi Laura, Marco Parrillo**  
> Università Telematica Internazionale UNINETTUNO, Faculty of Engineering – Computer Engineering

## Authors

- **Antonio Rocca** — Conceptualisation, methodology, software, formal analysis, investigation, writing – original draft, visualisation.
- **Luigi Laura** — Conceptualisation, supervision, writing – review & editing.
- **Marco Parrillo** — Data curation, validation, writing – review & editing.

## Overview

Synthetic seismic accelerogram generation is a relevant problem in earthquake engineering, especially when real strong-motion records are scarce for specific magnitude, distance, fault, or site-condition scenarios.

This project proposes a **Wavelet-Decomposed conditional GAN (WD-cGAN)** in which seismic accelerograms are decomposed into multiple wavelet sub-bands using the **Daubechies-4 discrete wavelet transform (db4 DWT)**. Each sub-band is evaluated by a dedicated discriminator, allowing the adversarial training process to focus separately on different frequency components of the seismic signal.

The current implementation conditions the generator on the earthquake moment magnitude \(M_w\).

## Key Features

- Conditional GAN architecture for seismic signal generation.
- Wavelet-domain decomposition using **db4 DWT**.
- Multi-discriminator adversarial training.
- One discriminator per wavelet sub-band.
- Energy-based weighting of discriminator contributions.
- Conditioning on moment magnitude \(M_w\).
- PyTorch implementation.
- Pre-processing pipeline for accelerometric records.
- Preliminary evaluation using spectral and wavelet-domain metrics.

## Proposed Architecture

The WD-cGAN architecture consists of:

1. A generator \(G\) that maps a latent vector \(z\), concatenated with a conditioning vector \(y\), into a synthetic accelerogram.
2. A DWT block that decomposes real and generated signals into wavelet sub-bands.
3. A set of parallel discriminators \(D_i\), each specialised on a specific wavelet component.
4. An energy-weighting module that assigns a coefficient \(\alpha_i\) to each discriminator according to the energy content of the corresponding sub-band.

The generator loss is obtained by aggregating the discriminator outputs through the energy-based weights:

\[
\mathcal{L}_G = \sum_{i=1}^{N} \alpha_i \mathcal{L}_{G,i}
\]

where \(N\) is the number of wavelet sub-bands.

In the present configuration, the model uses:

- wavelet: **Daubechies-4**
- decomposition level: **L = 6**
- wavelet components: **A6, D1, D2, D3, D4, D5, D6**
- number of discriminators: **N = 7**
- main conditioning variable: **moment magnitude \(M_w\)**

## Dataset

The experiments use accelerometric records from the **Italian Accelerometric Archive ITACA v4.0**, managed by the Istituto Nazionale di Geofisica e Vulcanologia — INGV.

Dataset source:

- ITACA v4.0: <https://itaca.mi.ingv.it>
- DOI: `10.13127/itaca.4.0`

The most significant training configuration used accelerograms from seismic events with:

\[
M_w \in [5.0, 5.5)
\]

After quality filtering, the dataset contained **1879 accelerograms**, sampled at:

\[
f_s = 200\,\mathrm{Hz}
\]

The accelerograms are normalised before training. Therefore, the current model output is dimensionless and must not be interpreted directly as physically calibrated acceleration unless a suitable amplitude-scaling procedure is applied.

## Repository Structure

```
Add
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/wd-cgan-seismic-accelerograms.git
cd wd-cgan-seismic-accelerograms
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Main Dependencies

The project requires:

```text
torch
numpy
scipy
pandas
matplotlib
PyWavelets
scikit-learn
tqdm
h5py
```

A minimal `requirements.txt` may contain:

```text
torch
numpy
scipy
pandas
matplotlib
PyWavelets
scikit-learn
tqdm
h5py
```

## Data Pre-processing

The pre-processing pipeline includes:

1. Loading ITACA accelerometric records.
2. Selecting the desired magnitude range.
3. Filtering records according to quality criteria.
4. Normalising each component to zero mean and unit variance.
5. Applying db4 DWT decomposition.
6. Computing the wavelet sub-band energy.
7. Building the conditioning vector \(y\).

Example command:

```bash
python src/preprocessing/load_itaca.py \
    --input data/raw \
    --output data/processed \
    --mw-min 5.0 \
    --mw-max 5.5 \
    --fs 200
```

## Training

To train the WD-cGAN model:

```bash
python src/training/train.py \
    --config configs/default.yaml
```

Example configuration:

```yaml
model:
  latent_dim: 400
  wavelet: db4
  decomposition_level: 6
  conditioning: Mw

training:
  epochs: 200
  batch_size: 16
  learning_rate: 0.0002
  beta1: 0.5
  beta2: 0.999

data:
  sampling_frequency: 200
  magnitude_min: 5.0
  magnitude_max: 5.5
```

## Generation

After training, synthetic accelerograms can be generated by specifying the conditioning magnitude:

```bash
python src/evaluation/generate.py \
    --checkpoint outputs/checkpoints/wdcgan.pt \
    --mw 5.3 \
    --num_samples 10 \
    --output outputs/generated
```

The generated signals are currently normalised. Physical amplitude calibration in \(\mathrm{cm/s^2}\) requires an external or integrated scaling procedure.

## Evaluation Metrics

The repository includes preliminary tools for evaluating synthetic accelerograms using:

- Power Spectral Density comparison.
- Per-sub-band wavelet energy ratio.
- Pearson correlation.
- Cross-correlation.
- Dynamic Time Warping.
- Kolmogorov-Smirnov distance.
- Wasserstein distance.
- PGA distribution checks.
- Arias intensity distribution checks.

## Current Limitations

The current implementation should be considered a research prototype.

Main limitations:

- Conditioning is currently limited to moment magnitude \(M_w\).
- The training dataset is geographically limited to Italian seismic sequences.
- The output accelerograms are normalised and not directly calibrated in physical units.
- Full validation on larger datasets is still required.
- Response-spectrum compatibility with engineering standards requires further testing.
- The generated accelerograms should not be used for operational engineering design without independent validation.

## Future Work

Planned developments include:

- Integration of larger datasets such as ESM and NGA-West2.
- Conditioning on additional physical parameters, including:
  - source-to-site distance \(R\),
  - site condition \(V_{S30}\),
  - fault mechanism,
  - hypocentral depth,
  - target PGA.
- GMPE-based amplitude scaling for physically calibrated accelerograms in \(\mathrm{cm/s^2}\).
- Architecture search over CNN, LSTM, GRU, and TCN discriminators.
- Physics-informed constraints for P/S-wave ordering and energy consistency.
- Validation against NTC2018 and Eurocode 8 response-spectrum compatibility criteria.
- Use in non-linear time-history structural analyses.

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{rocca2026wdcgan,
  author  = {Rocca, Antonio and Laura, Luigi and Parrillo, Marco},
  title   = {Synthetic Seismic Accelerogram Generation via Wavelet-Decomposed Conditional Generative Adversarial Networks},
  journal = {Journal Not Specified},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Related References

```bibtex
@article{florez2022data,
  author  = {Florez, M. A. and Caporale, M. and Buabthong, P. and Ross, Z. E. and Asimaki, D. and Meier, M.-A.},
  title   = {Data-driven synthesis of broadband earthquake ground motions using artificial intelligence},
  journal = {Bulletin of the Seismological Society of America},
  volume  = {112},
  number  = {4},
  pages   = {1979--1996},
  year    = {2022},
  doi     = {10.1785/0120210264}
}

@article{esfahani2023tfcgan,
  author  = {Esfahani, R. D. D. and Cotton, F. and Ohrnberger, M. and Scherbaum, F.},
  title   = {TFCGAN: Nonstationary ground-motion simulation in the time--frequency domain using conditional generative adversarial network (CGAN) and phase retrieval methods},
  journal = {Bulletin of the Seismological Society of America},
  volume  = {113},
  number  = {1},
  pages   = {453--467},
  year    = {2023},
  doi     = {10.1785/0120220068}
}

@article{matinfar2023dcgan,
  author  = {Matinfar, M. and Khaji, N. and Ahmadi, G.},
  title   = {Deep convolutional generative adversarial networks for the generation of numerous artificial spectrum-compatible earthquake accelerograms using a limited number of ground motion records},
  journal = {Computer-Aided Civil and Infrastructure Engineering},
  volume  = {38},
  number  = {2},
  pages   = {225--240},
  year    = {2023},
  doi     = {10.1111/mice.12852}
}
```

## Data Availability

The accelerometric records used in this study are publicly available from the Italian Accelerometric Archive ITACA v4.0, managed by INGV:

- <https://itaca.mi.ingv.it>
- DOI: `10.13127/itaca.4.0`

The repository does not redistribute the original ITACA records. Users should download the data directly from the official archive and comply with the corresponding data-use conditions.

## License

This repository is released for academic and research purposes.

Please check the `LICENSE` file before using the code or derived outputs in commercial, operational, or safety-critical applications.

## Disclaimer

The synthetic accelerograms generated by this model are intended for research purposes only.

They should not be used for engineering design, regulatory seismic verification, safety-critical structural assessment, or official hazard studies without independent validation, physical calibration, and compatibility checks against the applicable seismic codes and engineering standards.
