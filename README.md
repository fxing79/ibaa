# Intelligent Bayesian Asset Allocation

This repository provides a public available dataset to researchers and practitioners. They could experiment and benchmark their asset allocation models on it with/without sentiment information, or even use their own source of information. Please kindly cite the following paper if you find the dataset useful,  

Frank Xing, Erik Cambria, Lorenzo Malandri, and Carlo Vercellis (2018). [Discovering Bayesian Market Views for Intelligent Asset Allocation](https://link.springer.com/chapter/10.1007%2F978-3-030-10997-4_8). In Proceedings of ECML-PKDD, pp 120-135. [[pdf]](https://arxiv.org/pdf/1802.09911.pdf)
> Along with the advance of opinion mining techniques, public mood has been found to be a key element for stock market prediction. However, there has been little progress in leveraging public mood for the asset allocation problem. In order to address the issue of incorporating public mood analyzed from social media, we propose to formalize it into market views that can be integrated into the
modern portfolio theory. We train two neural models to generate the market views, benchmark the model performance using market views on other popular asset allocation strategies, and get some exciting results.

## Dataset Overview

The dataset comprises over 8 years of price data, trading volume data, and market capitalization data for the 5-stocks-portfolio experimented in the abovementioned paper. With **./mkt_cap** to calculate portfolio weights and **./price** data, one can easily replicate the numbers of **vw_pfl.txt**.

We are not authorized to publish sentiment data from **Psychsignal**, however, users could apply their own source of sentiment information.  
