## Preprocessing Note

For training, each accelerogram is cropped to a fixed-length symmetric time window centered on the time of peak ground acceleration (PGA).

If the PGA time is defined as

\[
t_{\mathrm{PGA}} = \arg\max_t |a(t)|,
\]

the input signal used for training is obtained as

\[
a_{\mathrm{win}}(\tau) = a(t_{\mathrm{PGA}}+\tau),
\qquad \tau \in [-T,T].
\]

This preprocessing step preserves the fixed-length input requirement of the GAN while reducing the influence of initial low-energy or non-significant signal portions. It also concentrates the learning process on the most informative part of the strong-motion record.
