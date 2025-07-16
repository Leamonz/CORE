# CORE: <u>C</u>arbon Di<u>O</u>xide <u>RE</u>construciton

## Abstract
Reconstructing atmospheric surface $\text{CO}_2$ is crucial for understanding climate dynamics and informing global mitigation strategies. Traditional inversion models achieve precise global $\text{CO}_2$ reconstruction but rely heavily on uncertain prior estimates of fluxes and emissions. Inspired by recent advances in data-driven weather forecasting, we explore whether data-driven models can reduce reliance on these priors. However, $\text{CO}_2$ reconstruction presents unique challenges, including complex spatio-temporal dynamics, periodic patterns and sparse observations. We propose $\text{CO}_2$-Net, a data-driven model that addresses these challenges without requiring extensive prior data. We formulate $\text{CO}_2$ reconstruction as solving a constrained advection-diffusion equation and derive three key components: physics-informed spatio-temporal factorization for capturing complex transport dynamics, wind-based embeddings for modeling periodic variations and a semi-supervised loss for integrating sparse $\text{CO}_2$ observations with dense meteorological data. $\text{CO}_2$-Net is designed in three sizes---small (S), base (B) and large (L)---to balance performance and efficiency. On CMIP6 reanalysis data, $\text{CO}_2$-Net (S) and (L) reduce RMSE by 11\% and 71\%, respectively, when compared to the best data-driven baseline. On real observations, $\text{CO}_2$-Net (L) achieves RMSE comparable to inversion models. The ablation study shows that the effectiveness of wind-based embedding and semi-supervised loss stems from their compatibility with our spatio-temporal factorization.

## Usage

## Cite

