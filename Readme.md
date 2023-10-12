# Hierarchical Explanations for Text Classification Models: Fast and Effective
The main implementation for the paper **Hierarchical Explanations for Text Classification Models: Fast and Effective**

## Requirement:

- pytoch == 1.9.1
- python == 3.7.11
- numpy == 1.21.2
- matplotlib == 3.4.3
- nltk == 3.6.5

## Model and data

The accuracy of the pre-trained model for each dataset is shown in Table2 of our paper.

Download the pre-trained models on each dataset  

After Download, you can put these files to the folder `Cla_datasets` and `TrainedModel` for the dataset and pre-trained models.

## Generate explanation

1.  Run the following command to generate explanations.

   `python HETSG_main.py --task_name sst-2 --start_pos 0 --end_pos -1`

The results of the explanation for each text will be saved in the folder `Explain_results`.

2. Run the following command to visualize the explanation on the given examples.

   `python HETSG_main.py --task_name sst-2 --visualize 1`
   

The figures of the hierarchical explanation will be saved in the main folder.


## Acknowledgement

Thanks for the following two repositories for their help in the implementation of our method.

https://github.com/zdgithub/Interpretable_Interaction_Trees#requirements

https://github.com/UVa-NLP/HEDGE#model-and-data