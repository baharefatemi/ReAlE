
Summary
=======

This repo contains the implementation of the model proposed in `Knowledge Hypergraph Embedding Meets Relational Algebra` for knowledge hypergraph embedding, as well as the code for synthesizing datasets as explained in the paper. 
The code can be also used to train a `ReAlE` model for any input graph. 


_Note however that the code is designed to handle graphs with arity at most 6 (which is the case for the datasets used in this paper)._

The software can be also used as a framework to implement new knowledge hypergraph embedding models.

## Dependencies

* `Python` version 3.7
* `Numpy` version 1.17
* `PyTorch` version 1.4.0

## Docker
We recommend running the code inside a `Docker` container. 
To do so, you will first need to have [Docker installed](https://docs.docker.com/).
You can then compile the image with:
```console
docker build -t hype-image:latest  .
```

and run using (replace the path to your local repo):
```console
docker run --rm -it -v {code-path}:/eai/project --user `id -u`:`id -g` hype-image /bin/bash
```

## Usage

To train HypE or any of the baselines you should define the parameters relevant to the given model.
The default values for most of these parameters are the ones that were used to obtain the results in the paper.

- `model`: The name of the model. Valid options are `RealE`, `HypE`, `HSimplE`, `MTransH`, `DistMult`, `MCP`.

- `dataset`: The dataset you want to run this model on (`JF17K` is included in this repo).

- `batch_size`: The training batch size.

- `num_iterations`: The total number of training iterations.

- `lr`: The learning rate.

- `nr`: number of negative examples per positive example for each arity.

- `window_size`: window size in ReAlE.

- `emb_dim`: embedding dimension.

- `non_linearity`: non-linearity function to apply on output of each window.

- `ent_non_linearity`: non-linearity to apply on entity embeddings before forwarding.

- `input_drop`: drop out rate for input layer of all models.

- `hidden_drop`: drop out rate for hidden layer of all models.

- `no_test_by_arity`: when set, test results will not be saved by arity, but as a whole. This generally makes testing faster. 

- `test`: when set, this will test a trained model on the test dataset. If this option is present, then you must specify the path to a trained model using `-pretrained` argument.

- `pretrained`: the path to a pretrained model file. If this path exists, then the code will load a pretrained model before starting the train or test process.
The filename is expected to have the form `model_*.chkpnt`. The directory containing this file is expected to also contain the optimizer as `opt_*.chkpnt`, if training is to resume. 

- `output_dir`: the path to the output directory where the results and models will be saved. If left empty, then a directory will be automatically created.

- `restartable`: when set, the training job will be restartable: it will load the model from the last saved checkpoint in `output_dir`, as well as the `best_model`, and resume training from that point on.
If this option is set, you must also specify `output_dir`.


## Cite ReAlE

If you use this package for published work, please cite the following paper:

      @misc{fatemi2021knowledge,
            title={Knowledge Hypergraph Embedding Meets Relational Algebra}, 
            author={Bahare Fatemi and Perouz Taslakian and David Vazquez and David Poole},
            year={2021},
            eprint={2102.09557},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
       }

Contact
=======

Bahare Fatemi

Computer Science Department

The University of British Columbia

201-2366 Main Mall, Vancouver, BC, Canada (V6T 1Z4)  

<bfatemi@cs.ubc.ca>


