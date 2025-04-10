# stpt2imc

[View Full Report (PDF)](./reports/Final_Report.pdf)

## Abstract

Serial Two-Photon Tomography (STPT) and Imaging Mass Spectrometry (IMC) are two popular imaging tech- niques in tumour analysis. STPT images describe tumour morphology, whereas IMC images describe protein abun- dance. Having both data modalities for the same tissue sample is often beneficial in clinical applications, as under- standing the tumour landscape from both a morphological and proteomic perspective can inform treatment decisions. However, it is difficult to obtain both data modalities for a single tissue sample. To mitigate this issue, we present a Generative Model that, for the same tissue sample, only re- quires STPT images to impute corresponding IMC images. 18 aligned STPT and IMC physical sections have been iden- tified and have been used for training two neural network models. The first model was based on the well known U- Net architecture, and the second model was based on a net- work designed for point clouds, called PointSetGen. The first model produced poor reconstruction results. The sec- ond model was not able to be trained because of hardware limitations and numerical instability and requires further attention.
