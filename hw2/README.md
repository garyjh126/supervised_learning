# Assignment Two

You may work either alone or in teams of two.

## Task: supervised learning, get good generalisation.

We have numeric input data, 8 dimensional, and binary classification.

## You are given:

* File `train.txt`, a training set of 100,000 labelled inputs, roughly balanced between classes. Each line is one input-output pair, 9 numbers/line; the first 8 are the input, the last is the label (0/1).

* File `test.txt`, a test set of 10,000 unlabelled inputs.  Each line is one input, 8 numbers/line.

## Other details

All the inputs, both training and testing, were drawn iid from the same distribution.

You can use whatever software and machine learning algorithms and methods you wish (within reason, e.g., don't hire someone to do this for you, or use some totally-automated ML-homework-solving system; if you're not sure ask).  Examples: `mlpack`, Python `scikit-learn`, Octave and its Machine Learning packages, R and its cran-xxx Machine Learning packages, `pytorch`, etc.

## Turn in (via moodle)

Only one member of each team should turn in the assignment.

1. Be sure to put the **names of all team members** prominently on the front of the report. I will assume that all team members have approved all of the materials turned in.

2. Label the test set: a plain ascii file `test-labels.txt` of 10,000 labels, each 0 or 1, one per line, same order as in `test.txt`. The more of these you get right (above chance performance), the higher your score on this assignment.

3. Include the code you wrote and associates methods you used, along with a pointer to any libraries or other relevant software.  Ideally, I should be able to type `make clean; make test-labels.txt` and have the system train from scratch on `train.txt` and then read `test.txt` and generate `test-labels.txt`. Or `jupyter notebook hw2.ipy` and hit the *run* button.

4. Include a brief report (ideally a PDF file named `hw2.pdf` or `hw2.md` with inline images) explaining what you did.  Include graphs showing cross-validation performance that led you to the architecture you settled on, etc.

5. For *OPTIONAL EXTRA CREDIT*.  There is hidden structure in the data!  Find it and (in your report) show me what it is and how you found it.

6. For team projects: please include a *brief* description of each person's role on the project as the first paragraph of the report.
