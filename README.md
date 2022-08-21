This code is for "Identification of personalized driver genes for individuals using graph convolution network"


pDriverGCN should be run under the environment of python >=3.7 .
- Operation steps:
- 1.Run data_process.py LUAD(cancer data folder)
	- The output of data_process.py is 4 files after preprocessing: exp.txt, mut.txt, PPI.txt, mut_driver.txt.
- 2.Run train.py LUAD
	- The output of train.py is a matrix of gene score of samples
- 3.Run deal_result.py LUAD
- 4.Run condorcet_R.py LUAD
	- The output of condorcet_R.py is gene ranking.
