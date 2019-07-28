# GSARF: State Adaptive Random Forest on GPU

Random Forests is a classical ensemble method used to improve the performance of single tree classifiers.  In evolving data streams, the classifier has also to be adaptive and work under very strict constraints of space and time. The computational load of using a large number of classifiers can make its application extremely expensive. One clear benefit of random forest is its ability to be executed in parallel. Most research concentrates on heuristics that optimize the code, fundamental approaches for fast execution of learned models based on computational architecture is rare. In our research we introduce a random forest model utilizing both GPU and CPU, called State-Adaptive Random Forest on GPU (GSARF).  We address the pre-existing challenges of adapting random forest for data streams, specifically in the area of continual learning, whereby we reuse trees in the random forest when old concepts reappear. Current random forest in data streams stores two types of trees, foreground trees which are trees that are currently used in prediction, background trees which are trees that are built when we are aware of possible changes in the data streams. Additionally, we store candidate trees, which are trees that had been highly used in the previous concepts, but are now discarded due to changes in the data stream. We store these trees in a repository as they may become relevant at some future point and can be reused. By having this repository, we can reduce computation cost. Our approach has shown to outperform a baseline GPU-based approach in terms of accuracy performance. 


### Prerequisites

CUDA Toolkit 10.0

### Installing

Compile by running

```
make
```

### Data Preparation

Covtype is already included under `data/covtype/`. All the other datasets need to be generated
and binned by the following instructions:

##### Synthetic Data generation

Generate Agrawal datasets
```
cd ./moa/agrawal/
./agrawal_data_generator.sh
```

Generate LED datasets
```
cd ./moa/led/
./led_data_generator.sh
```

##### Real Dataset Preparation

KDD99 can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). Download and
decompress it under `data/kddcup/`, then bin it by
running the following script:

```
./data/kddcup/convert.py
```


## Running the Tests

Running GARF and GSARF
```
./agrawal_batch_run.sh
./led_batch_run.sh
./covtype.sh
./kddcup.sh
```

Running ARF (MOA)
```
cd ./moa/agrawal/
./moa_arf_runner.sh
```

```
cd ./moa/led/
./moa_arf_runner.sh
```

Results of all the tests and scripts for interpreting the results can be found under each of the
dataset folder i.e. `data/[dataset_name]/`.

#### Interpreting the Test Results


```
cd data/[dataset_name]/ 
```

Draw accuracy comparison chart between GARF and GSARF.
```
gnuplot -p draw.p
```

Draw gain chart.
```
gnuplot -p draw_gain.p
```

Evaluate memory consumption on host machine.
```
./mem_eval.sh
```

Calculate mean and standard deviation for accuracy on the 10 seeds.
```
./stats_eval.sh
```

Calculate mean and standard deviation for time on the 10 seeds.
```
./time_eval.sh
```

#### Options

```
  -c turns on the state-adaptive algorithm
  -t number of trees
  -i number of instances to be trained at the same time on GPU
  -d depth of the trees [1, 11]
  -e maximum edit distance
  -k kappa threshold
  -x drift warning threshold
  -y drift threshold
  -b delta threshold for adding background tree to the CPU tree pool
  -p required. Path that contains the data file
  -n required. Data file name
```


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />

This project is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

Please contact us for commercial usages.
