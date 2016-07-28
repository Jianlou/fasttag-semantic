
1. download the 15 descriptors for all the three datasets at
http://lear.inrialpes.fr/people/guillaumin/data.php

2. update the mainFolder in run.m to the correct location of the descriptors

(Due to the random projection used in the preprocessing step, the final
numbers could differ slightly. To reproduce the numbers in the log file, you
could instead skip the first two steps, and download the preprocessed data at 
http://www.cse.wustl.edu/~mchen/code/fastTag/data/
)

3. the regularization factor beta might need to be tuned for other datasets

4. start matlab, and run 
run.m

Please contact chenmm24@gmail.com if you have any questions. 
