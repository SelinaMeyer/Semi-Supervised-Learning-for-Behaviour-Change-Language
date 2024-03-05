This is the repository for the paper **Exploring the Challenges of Behaviour Change Language Classification: A Study on Semi-Supervised Learning and the Impact of Pseudo-Labelled Data**

The GLoHBCD, used in the paper for baseline experiments, can be replicated here: https://github.com/SelinaMeyer/GLoHBCD

The data used for pseudo-labelling can be obtained as follows:
AnnoMI is publicly available: https://github.com/uccollab/AnnoMI
The new forum data used for pseudo-labelling can be obtained with the help of the Crawlers provided in the ``Crawler`` folder.

Prefilter.ipynb was used as the relevance filter (Stage 1 of experiments)
Stratified_Fine_tuning.ipynb was used for obtaining confidence thresholds in Stage 1 and conduct Experiments in Stages 2 and 3
Final_finetuning was used to conduct experiments in Stage 4
