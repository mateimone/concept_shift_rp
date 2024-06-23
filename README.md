# A Benchmark of Concept Shift Impact on Federated Learning Models.

This repository contains the necessary code to run the experiments for the **A Benchmark of Concept Shift Impact on Federated Learning Models - Comparing the differences in performance between federated and centralized models under concept shift** research project.
This project has been produced as part of CSE3000 - Research Project, at Delft University of Technology, The Netherlands.

**Author**: Matei Ivan Tudor (m.t.ivan-1@student.tudelft.nl)

Firstly, install the dependencies in requirements.txt. Make sure to install torch and torchvision with CUDA enabled, if your machine is able to perform calculations on CUDA cores.

To run the experiments:
1) CIFAR-10 : in the cifar_classes package, run the cifar_centralized.py and cifar_federated.py classes to train a centralized/federated model on CIFAR-10. Then, run the class get_model_accuracies.py in order to obtain the accuracies of the models on the test set, with your choice of transforms.
2) River 2D Dataset : in the hyperplane_classes package, run the hyperplane_mlp_training_centralized.py and hyperplane_mlp_training_federated.py classes to train a centralized/federated model for a tabular dataset of your choice (datasets can be found in the hyperplanes and hyperplanes_with_redundancy folders). Run the accuracy_over_rotations.py class in order to get the accuracies of the saved models. We already provide hyperplanes, but if you want to generate different ones, you can find how to create them [here](https://riverml.xyz/0.21.1/api/datasets/synth/Hyperplane/). Then, to rotate the hyperplane simply make use of the rotate_hyperplane.py class, and to add redundant features, make use of the add_redundant_features.py class. 
3) Lastly, to see the JS calculations, run the js_distance.py class. To see the cumulative variance for principal components calculations, run the principle_components.py class. Both are found in the src package. 

