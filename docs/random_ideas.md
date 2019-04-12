1) Principal Component Analysis (selected 150 components 95% variance).
	The following classifiers didn't seem to do well at all
2) Tried simple classifiers
	a) Logistic Regression
	b) Linear SVM
	c) Kernel SVM (RBF)
	d) Decision Trees (I forgot this)
	e) kNN
	f) Random Forests
3) Implemented feature selection using FEAST.
4) Also try XGBoost?
5) Logistic regression with L1, L2, elastic net regularization?
6) Selected patients so the classes are balanced
7) Question about double dipping (feature selection and then principal component analysis)
8) Picking RBF SVM based on initial assessment on F1 plots and need to tune the hyperparameters
9) Network rows:

	a. AUD 0 - 2
	b. CB 3 - 4
	c. CC 5 - 11
	d. DMN 12 - 26
	e. S 27
	f. SC 28 - 29
	g. SM 30 - 34
	h. VIS 35 - 45

Idea 01: Hierarchical modeling on the individual domains (question from victor: how to account for interactions)

Idea 02: Fitting a classifier on the dfnc windows across subjects (for each time point) and then doing an argmax on the 
predicted labels from each classifier

Idea 03: Fitting a classifier on the dfnc windows across subjects (for each time point) and then stack the predicted labels.
Fit another classifer on the stacked dataset and that will be final predicted labels. 

Idea 04 (somewhat orthogonal): Split HD group in half based on Langbehn model and build two prediction models (would establish approx. balanced groups) --> Mason et al. (Annals of Neurology, 2018): "The preHD group was median-split into preHD-near and preHD-far subgroups according to their estimated years to clin- ical diagnosis score calculated using the Langbehn model (CAG repeat length and age). 

Discussion on 04/11/2019:
Victor wants 
1) To compare HC and Early/Low/Near (based on cap_d_Scores). The expectation is no significant differences
resulting in a classifier performance of ~50%.
2) To compare HC and High (based on cap_d_score). The expectation is there will be significant differences resulting in a higher accuracy.

