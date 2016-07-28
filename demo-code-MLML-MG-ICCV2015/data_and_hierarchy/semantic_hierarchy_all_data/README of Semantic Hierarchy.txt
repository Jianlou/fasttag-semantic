This folder includes all information about the semantic hierarchies for the four datasets used in the paper, including 
'corel5k_hierarchy_structure.mat',
'espgame_hierarchy_structure.mat',
'iaprtc12_hierarchy_structure.mat',
'mediamill_hierarchy_structure.mat'. 

For each structure, there are 6 fields:
edgeMatrix -- n_e x 2 matrix, in each row, there are two entries as (child, parent) class
parent_matrix --  n_e x m matrix, one row corresponds one child class, only its parent classes are 1 in this row
ancestor_matrix --  n_e x m matrix, one row corresponds one child class, only its ancestor classes are 1 in this row
label_train_full --  m x n_sample_train sparse matrix, the filled-in train label matrix by semantic hierarchy 
label_test_full --  m x n_sample_test sparse matrix, the filled-in test label matrix by semantic hierarchy 
label_name = ids_corel5k; % the name of all classes

You can load the structure and display the semantic hierarchy in matlab, using the following codes:
-> load('corel5k_hierarchy_structure.mat')
-> parent_matrix = corel5k_hierarchy_structure.parent_matrix; 
-> label_name_corel5k = corel5k_hierarchy_structure.label_name; 
-> view(biograph(parent_matrix',label_name_corel5k))
